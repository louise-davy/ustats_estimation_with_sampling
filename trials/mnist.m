% run MNIST experiment
	
function [] = mnist(b, mode, n2, eta0, maxIter, valSize, testSize, seed)
	
    b = str2num(b)
	n2 = str2num(n2)
	eta0 = str2num(eta0)
	maxIter = str2num(maxIter)
	testSize = str2num(testSize)
	valSize = str2num(valSize)
	seed = str2num(seed)
	mode
	iterRisk=[1:10:100 100:100:1000 1000:1000:10000]
	
	load /cluster/fhgfs/abellet/MNIST/data/mnist_164_unitnorm;
	
	% set random number generator to fixed number to generate val and test pairs
	rng(11111);
	
	% draw random pairs as the val set and precompute some quantities
	nTrain = size(XTr,1);
	pairsVal = zeros(valSize,2);
	pairsVal(:,1) = randsample(nTrain,valSize,true);
	pairsVal(:,2) = randsample(nTrain,valSize,true);
	pYVal = -ones(valSize,1);
	pYVal(YTr(pairsVal(:,1))==YTr(pairsVal(:,2))) = 1;
	diffVal = XTr(pairsVal(:,1),:)-XTr(pairsVal(:,2),:);
	clear pairsVal;
	
	% likewise, draw random pairs as the test set and precompute some quantities
	nTest = size(XTe,1);
	pairsTest = zeros(testSize,2);
	pairsTest(:,1) = randsample(nTest,testSize,true);
	pairsTest(:,2) = randsample(nTest,testSize,true);
	pYTe = -ones(testSize,1);
	pYTe(YTe(pairsTest(:,1))==YTe(pairsTest(:,2))) = 1;
	diffTe = XTe(pairsTest(:,1),:)-XTe(pairsTest(:,2),:);
	clear pairsTest XTe YTe;
	
	saveDir = ['/cluster/fhgfs/abellet/MNIST/res_MNIST2/bias' num2str(b) '_mode' mode '_n2' int2str(n2) '_eta0' num2str(eta0) '_maxIter' int2str(maxIter) '_valSize' int2str(valSize) '_testSize' int2str(testSize) '.' int2str(seed)]
	mkdir(saveDir);
	
	% initialize random number generator with seed for reproducibility
	% but do not choose same seed for i and p to avoid correlation
	if mode == 'p'
		rng(seed);
	elseif mode == 'i'
		rng(seed+1000);
	else
		error('parameter mode invalid');
	end
		
	% run algorithm	
	sgd_pairs(XTr, YTr, diffVal, pYVal, diffTe, pYTe, b, mode,  n2, eta0, maxIter, iterRisk, saveDir);
	
end


% SGD algorithm for metric learning with hinge loss objective
% loss is max( 0 , 1+y(d(x_i,x_j)-b) )
% X: n x d matrix (training data)
% Y: n x 1 vector (training labels)
% diffVal: val pairs
% pYVal: val pair labels
% diffTe: test pairs
% pYTe: test pair labels
% b: 1 x 1 (offset in hinge loss)
% mode: 'p' for pair-based sampling, 'i' for instance-based sampling
% n2: size of u-stat used for mini-batch (number of pairs is nchoosek(n2,2))
% eta0: initial step size
% maxIter: max number of iterations
% iterRisk: iterations where the risk is computed and the iterate is saved
% saveDir: directory to save results

function res = sgd_pairs(X, Y, diffVal, pYVal, diffTe, pYTe, b, mode,  n2, eta0, maxIter, iterRisk, saveDir)

	n = size(X,1);
	d = size(X,2);
	M = eye(d); % initialize M to identity matrix
	
	countRisk=1;

	% number of pairs in mini-batch
	m = nchoosek(n2,2);
	
	for k=1:maxIter
		
		if mode == 'p'
			% sample m pair indices with replacement
			% note that an instance can be paired with itself
			% but effect is negligible when n is large
			pairs = zeros(m,2);
			pairs(:,1) = randsample(n,m,true);
			pairs(:,2) = randsample(n,m,true);
		elseif mode == 'i'
			% sample instances with replacement such that we have m pairs
			s = randsample(n,n2,true);
			% take all combinations
			pairs = nchoosek(s,2);
		end

		% compute pair labels
		pY = -ones(m,1);
		pY(Y(pairs(:,1))==Y(pairs(:,2))) = 1;

		gradM = zeros(d,d);
		diff = X(pairs(:,1),:)-X(pairs(:,2),:);
		slack = (1+pY'.* (sum((M*diff').*diff' , 1) - b)) > 0;
		for i=1:m % for each pair
			if slack(i)
				gradM = gradM + pY(i)*diff(i,:)'*diff(i,:);
			end
		end
		gradM = gradM/m;
		
		% update with step size decreasing in 1/k
		M = M - eta0*(1/k)*gradM;

		if ismember(k,iterRisk)
			% project onto the PSD cone
			% one-time projection paradigm
			[V, L] = eig(M);
			V = real(V); L = real(L);
			ind = find(diag(L) > 0);
			M2 = V(:,ind) * L(ind, ind) * V(:,ind)';
			% compute val risk
			RVal = compute_risk_pairs(diffVal, pYVal, b, M2);
			% compute test risk
			RTe = compute_risk_pairs(diffTe, pYTe, b, M2);
			res(countRisk,:) = [k RVal RTe];
			fprintf('iter %d, val risk %g, test risk %g\n',k,RVal,RTe);
			countRisk = countRisk+1;
			save([saveDir '/M.' int2str(k) '.mat'],'M2');
		end
			
	end
	
	save([saveDir '/res.mat'],'res');

end

% compute value of risk on pairs
function R = compute_risk_pairs(diffTe, pYTe, b, M)
	slack = max(0, 1+pYTe'.* (sum((M*diffTe').*diffTe' , 1) - b));
	R = mean(slack);
end  
