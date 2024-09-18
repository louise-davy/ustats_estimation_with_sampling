% retrieve MNIST experiment results

function [] = make_res2(b, n2, eta0, maxIter, valSize, testSize, runs)

	path = ['res_MNIST2/bias' num2str(b) '_modei_n2' int2str(n2) '_eta0' num2str(eta0) '_maxIter' int2str(maxIter) '_valSize' int2str(valSize) '_testSize' int2str(testSize) '.1/res.mat'];
	load(path);
	resITr = res(:,1:2);
	resITe = res(:,[1 3]);
	for s=2:runs
		path = ['res_MNIST2/bias' num2str(b) '_modei_n2' int2str(n2) '_eta0' num2str(eta0) '_maxIter' int2str(maxIter) '_valSize' int2str(valSize) '_testSize' int2str(testSize) '.' int2str(s) '/res.mat'];
		load(path);
		resITr(:,s+1) = res(:,2);
		resITe(:,s+1) = res(:,3);
	end
	
	path = ['res_MNIST2/bias' num2str(b) '_modep_n2' int2str(n2) '_eta0' num2str(eta0) '_maxIter' int2str(maxIter) '_valSize' int2str(valSize) '_testSize' int2str(testSize) '.1/res.mat'];
	load(path);
	resPTr = res(:,1:2);
	resPTe = res(:,[1 3]);
	for s=2:runs
		path = ['res_MNIST2/bias' num2str(b) '_modep_n2' int2str(n2) '_eta0' num2str(eta0) '_maxIter' int2str(maxIter) '_valSize' int2str(valSize) '_testSize' int2str(testSize) '.' int2str(s) '/res.mat'];
		load(path);
		resPTr(:,s+1) = res(:,2);
		resPTe(:,s+1) = res(:,3);
	end
	
	clear res;
	
	meanITr = mean(resITr(:,2:runs+1),2);
	meanITe = mean(resITe(:,2:runs+1),2);
	meanPTr = mean(resPTr(:,2:runs+1),2);
	meanPTe = mean(resPTe(:,2:runs+1),2);
	stdITr = std(resITr(:,2:runs+1),1,2);
	stdITe = std(resITe(:,2:runs+1),1,2);
	stdPTr = std(resPTr(:,2:runs+1),1,2);
	stdPTe = std(resPTe(:,2:runs+1),1,2);
	
	plot(resPTr(:,1),meanPTr,'b','LineWidth',3);
	hold on;
	plot(resPTr(:,1),meanPTr+stdPTr,'b','LineWidth',1);
	plot(resPTr(:,1),meanPTr-stdPTr,'b','LineWidth',1);
	plot(resITr(:,1),meanITr,'r','LineWidth',3);
	plot(resITr(:,1),meanITr+stdITr,'r','LineWidth',1);
	plot(resITr(:,1),meanITr-stdITr,'r','LineWidth',1);
	hold off;
	
	savePath = ['res_MNIST2/plot_bias' num2str(b) '_n2' int2str(n2) '_eta0' num2str(eta0) '_maxIter' int2str(maxIter) '_valSize' int2str(valSize) '_testSize' int2str(testSize) '_runs' int2str(runs) '_meanstdTr.eps'];
	saveas(gca, savePath, 'epsc');
	
	clf;

	plot(resPTe(:,1),meanPTe,'b','LineWidth',3);
	hold on;
	plot(resPTe(:,1),meanPTe+stdPTe,'b','LineWidth',1);
	plot(resPTe(:,1),meanPTe-stdPTe,'b','LineWidth',1);
	plot(resITe(:,1),meanITe,'r','LineWidth',3);
	plot(resITe(:,1),meanITe+stdITe,'r','LineWidth',1);
	plot(resITe(:,1),meanITe-stdITe,'r','LineWidth',1);
	hold off;
	
	savePath = ['res_MNIST2/plot_bias' num2str(b) '_n2' int2str(n2) '_eta0' num2str(eta0) '_maxIter' int2str(maxIter) '_valSize' int2str(valSize) '_testSize' int2str(testSize) '_runs' int2str(runs) '_meanstdTe.eps'];
	saveas(gca, savePath, 'epsc');
	
	clf;
	
	hold on;
	for s=1:runs
		plot(resPTr(:,1),resPTr(:,s+1),'b','LineWidth',1);
		plot(resITr(:,1),resITr(:,s+1),'r','LineWidth',1);
	end
	hold off;
	
	savePath = ['res_MNIST2/plot_bias' num2str(b) '_n2' int2str(n2) '_eta0' num2str(eta0) '_maxIter' int2str(maxIter) '_valSize' int2str(valSize) '_testSize' int2str(testSize) '_runs' int2str(runs) '_indTr.eps'];
	saveas(gca, savePath, 'epsc');

	clf;

	hold on;
	for s=1:runs
	plot(resPTe(:,1),resPTe(:,s+1),'b','LineWidth',1);
	plot(resITe(:,1),resITe(:,s+1),'r','LineWidth',1);
	end
	hold off;
	
	savePath = ['res_MNIST2/plot_bias' num2str(b) '_n2' int2str(n2) '_eta0' num2str(eta0) '_maxIter' int2str(maxIter) '_valSize' int2str(valSize) '_testSize' int2str(testSize) '_runs' int2str(runs) '_indTe.eps'];
	saveas(gca, savePath, 'epsc');
		
end
