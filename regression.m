clear
load mnist.mat
load regressionnet.mat
%% 
%selecting and saving images to train
numE = 1600;
validP = 0.25;
idx1 = (training.labels == 1);
place1 = find(cumsum(idx1) > numE-1);
idx2 = (training.labels == 2);
place2 = find(cumsum(idx2) > numE-1);
idx4 = (training.labels == 4);
place4 = find(cumsum(idx4) > numE-1);
idx8 = (training.labels == 8);
place8 = find(cumsum(idx8) > numE-1);
images1 = training.images(:,:,idx1(1:place1));
images2 = training.images(:,:,idx2(1:place2));
images4 = training.images(:,:,idx4(1:place4));
images8 = training.images(:,:,idx8(1:place8));
trainingImages = cat(3, images1(:,:,1:(numE*(1-validP))), images2(:,:,1:numE*(1-validP)), images4(:,:,1:numE*(1-validP)), images8(:,:,1:numE*(1-validP)));
validationImages = cat(3, images1(:,:,(numE*(1-validP)+1):numE), images2(:,:,(numE*(1-validP)+1):numE), images4(:,:,(numE*(1-validP)+1):numE), images8(:,:,(numE*(1-validP)+1):numE));

delete('trainingGround\*')
for k = 1:size(trainingImages, 3)
    imwrite(trainingImages(:,:,k),"trainingGround\image"+num2str(k)+".png");
end

delete('validationGround\*')
for k = 1:size(validationImages, 3)
    imwrite(validationImages(:,:,k),"validationGround\image"+num2str(k)+".png");
end

trainingImagesNoised = trainingImages - randn(28)/2;
delete('trainingImages\*')
for k = 1:size(trainingImagesNoised, 3)
    imwrite(trainingImagesNoised(:,:,k),"trainingImages\image"+num2str(k)+".png");
end

validationImagesNoised = validationImages - randn(28)/2;
delete('validationImages\*')
for k = 1:size(validationImagesNoised, 3)
imwrite(validationImagesNoised(:,:,k),"validationImages\image"+num2str(k)+".png");
end

%creating imagedatastore from the files
ds = imageDatastore("trainingGround\");
valDs = imageDatastore("validationGround\");
noisedDs = imageDatastore("trainingImages\");
noisedValDs= imageDatastore("validationImages\");

combinedDs = combine(noisedDs, ds);
combinedValDs = combine(noisedValDs, valDs);
%% 
%setting training options
options = trainingOptions( ...
'adam',...
'MiniBatchSize', 4800,...
'MaxEpochs',3000, ...
'Plots', 'training-progress', ...
'ValidationData', combinedValDs, ...
'ValidationFrequency',20, ...
'ValidationPatience',15);
%% 
%training net
net =  trainNetwork(combinedDs, lgraph_4, options);
save((num2str(numE*3))+"_images_trained_net", "net")
%% 
%test code
for k=1:8
    idx = 1;
    trueCheck = 0;
    while trueCheck ~= 1
    idx = randi(size(test.images,3));
    if test.labels(idx) == (6)
        trueCheck = 1;
    end  
    end
    subplot(2,8,k)
    noisedTest = test.images(:,:,idx) - randn(28)/2;
    imagesc(noisedTest*255)
    colormap(gray)
    title("ground "+num2str(idx))
    subplot(2,8,k+8)
    prediction = predict(net,noisedTest*255);
    imagesc(prediction(:,:,1))
    colormap(gray)
    title("prediction "+num2str(idx))
end
%%
for k=1:8
    idx = randi(size(trainingImagesNoised,3));
    subplot(2,8,k)
    imagesc(trainingImagesNoised(:,:,idx)*255)
    colormap(gray)
    title("ground "+num2str(idx))
    subplot(2,8,k+8)
    prediction = predict(net,trainingImagesNoised(:,:,idx)*255);
    imagesc(prediction(:,:,1))
    colormap(gray)
    title("prediction "+num2str(idx))
end
