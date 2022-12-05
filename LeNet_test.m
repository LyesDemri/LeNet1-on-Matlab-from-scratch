clear;clc;close all;
%This program tests a trained LeNet network on the MNIST validation images.
%It's just the forward pass of the training script with the counting of
%errors.
%Open images:
load('t10k-images.idx3-ubyte.mat');
load('t10k-labels.idx1-ubyte.mat');
pixel = pixel/255;

load('LeNet.mat');
errors = 0;
for image = 1:10000
    I = pixel(:,:,image);
    correct_answer = label(image);

	for f = 1:4
        H{1}{f} = conv_filter(I,Ws{1}{f},bs{1}{f});
    end

    %first subsampling layer:
    for f = 1:4
        H{2}{f} = subsampling_filter(H{1}{f},Ws{2}{f});
    end
    %second convolutional layer:
    H{3}{1} = H{2}{1};
    H{3}{2} = H{2}{1} + H{2}{2};
    H{3}{3} = H{2}{1} + H{2}{2};
    H{3}{4} = H{2}{2};
    H{3}{5} = H{2}{1} + H{2}{3};
    H{3}{6} = H{2}{1} + H{2}{3};
    H{3}{7} = H{2}{3};
    H{3}{8} = H{2}{3} + H{2}{4};
    H{3}{9} = H{2}{3} + H{2}{4};
    H{3}{10} = H{2}{4};
    H{3}{11} = H{2}{3} + H{2}{4};
    H{3}{12} = H{2}{3} + H{2}{4};

    for f = 1:12
        H{4}{f} = conv_filter(H{3}{f},Ws{3}{f},bs{3}{f});
    end

    %second subsampling layer:
    for f = 1:12
        H{5}{f} = subsampling_filter(H{4}{f},Ws{4}{f});
    end

    n = 1;
    for f = 1:size(H{5},2)
        X(n:n+numel(H{5}{f})-1,1) = reshape(H{5}{f},16,1);
        n=n+numel(H{5}{f});
    end
    final_output = logsig(Wfc*X+bfc);
    [osf,given_answer] = max(final_output);
    if (given_answer-1 ~= label(image))
        errors = errors+1;
    end
    disp(['Image ' num2str(image) ', errors: ' num2str(errors) ', error rate: ' num2str(errors*100/image)]);
end
disp(['Number of errors: ' num2str(errors)])
disp(['Error rate: ' num2str(errors*100/10000)])
disp(['Correct recognition rate: ' num2str(100 - errors*100/10000)])
%this next line is a very lazy way of keeping track of the correct
%recognition rates for each epoch. You can do much much better. It basically
%writes the correct recognition rate as the name of a .mat file. Terrible.
save(['LeNet_' num2str(num2str(100 - errors*100/10000)) '.mat']);

%You can uncomment this line to have the training script be called
%immediately after this test script.
%LeNet_train
%If you uncomment the last line of the training script as well, you can
%have the scripts call each other indefinitely (you can stop them with
%CTRL+C). It's useful if you want to leave your computer on and let it
%train overnight or something. My laptop was able to do 4 epochs in one
%night.