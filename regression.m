clear
load mnist.mat
load regressionnet.mat
%%
%selecting and saving images to train
idx = 0;
numE = 300;
validP = 0.10;
testP = 0.20;
trainingImages = [];
validationImages = [];
testImages = [];
for i=0:9
idx = (training.labels == i);
place = find(cumsum(idx) > numE-1);
images = training.images(:,:,idx(1:place));
trainingImages = cat(3, trainingImages, images(:,:,1:(numE*(1-validP))));
validationImages = cat(3, validationImages, images(:,:,(numE*(1-validP)+1:size(images,3))));
idxTest = (test.labels == i);
placeTest = find(cumsum(idxTest) > numE*testP-1);
imagesTest = test.images(:,:,idxTest(1:placeTest));
testImages = cat(3, testImages, imagesTest);
end

delete('trainingGround\*')
for k = 1:size(trainingImages, 3)
    imwrite(trainingImages(:,:,k),"trainingGround\image"+num2str(k)+".jpg");
end

delete('validationGround\*')
for k = 1:size(validationImages, 3)
    imwrite(validationImages(:,:,k),"validationGround\image"+num2str(k)+".jpg");
end

delete('testGround\*')
for k = 1:size(testImages, 3)
    imwrite(testImages(:,:,k),"testGround\image"+num2str(k)+".jpg");
end

trainingImagesNoised = rescale(trainingImages - (randn(28)));
delete('trainingImages\*')
for k = 1:size(trainingImagesNoised, 3)
    imwrite(trainingImagesNoised(:,:,k),"trainingImages\image"+num2str(k)+".jpg");
end

validationImagesNoised = rescale(validationImages - (randn(28)));
delete('validationImages\*')
for k = 1:size(validationImagesNoised, 3)
imwrite(validationImagesNoised(:,:,k),"validationImages\image"+num2str(k)+".jpg");
end

testImagesNoised = rescale(testImages - (randn(28)));
delete('testImages\*')
for k = 1:size(testImagesNoised, 3)
imwrite(testImagesNoised(:,:,k),"testImages\image"+num2str(k)+".jpg");
end


%creating imagedatastore from the files
ds = imageDatastore("trainingGround\");
valDs = imageDatastore("validationGround\");
noisedDs = imageDatastore("trainingImages\");
noisedValDs= imageDatastore("validationImages\");
testDs = imageDatastore("testGround\");
noisedTestDs = imageDatastore("testImages\");

combinedDs = combine(noisedDs, ds);
combinedValDs = combine(noisedValDs, valDs);
combinedTestDs = combine(noisedTestDs, testDs);
%% 
%setting training options
options = trainingOptions( ...
'adam',...
'MiniBatchSize', 2700,...
'MaxEpochs',3000, ...
'Plots', 'training-progress', ...
'ValidationData', combinedValDs, ...
'ValidationFrequency',20, ...
'ValidationPatience',1);
%% 
%training net
net =  trainNetwork(combinedDs, lgraph_1, options);
save((num2str(numE*3))+"_images_trained_net", "net")
%% 
%prediction
for k=1:8  
    idx = randi(size(testImages,3));
    subplot(2,8,k)
    image(readimage((noisedTestDs),idx));
    colormap(gray)
    title("ground "+num2str(idx))
    subplot(2,8,k+8)
    prediction = predict(net,noisedTestDs);
    image(prediction(:,:,1,idx))
    colormap(gray)
    title("prediction "+num2str(idx))
end
%% 
% %loss calc
% totalL = 0;
% for i=1:size(trainingImagesNoised,3)
%     prediction = predict(net, noisedTestDs);
%     sumL = 0;
%     for n=1:size(prediction,1)
%         rowsum = 0;
%         for m=1:size(prediction,2)
%         g = trainingImages(n,m,i);
%         p = prediction(n,m);
%         pixelL = (g-p)^2;
%         rowsum = rowsum + pixelL;
%         end
%         sumL = sumL + rowsum;
%     end
%     totalL = totalL + sumL;
%     Loss = sqrt(totalL/(28*28));
% end


