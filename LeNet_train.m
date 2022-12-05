clear;clc;close all;
tic
%This program reproduces the network described in LeCun 1989 (Handwritten 
%Digit Recognition with a Back-Propagation Network)

%Open image:
load('train-images.idx3-ubyte.mat');
load('train-labels.idx1-ubyte.mat');
pixel = pixel/255;

L = 4;  %number of layers
%NB: there's a fifth layer between the 3rd and 4th that sums together the
%subsampled images of layer 2.
layerTypes = {'conv','subsampling','conv','subsampling'};
numFilters = [4,4,12,12];   %number of filters (neurons) for each layer
w = 5;  %filter width and height

%I've set up the program so that you can either start a new network from
%epoch 1 or add an epoch to a pre-existing network. set new_network to
%false if you want to add an epoch
new_network = true;
if new_network
    %initialize network:
    for l=1:L
    	%for convolutional layers, generate random values
        %for subsampling layers, generate values that will average
        %the 4 input pixels.
        if strcmp(layerTypes{l},'conv')
            for f = 1:numFilters(l)
                W{f} = rand(w,w)*2-1;
                %Another LeCun paper mentions that biases aren't shared
                %so for each filter we have a matrix of biases that is the
                %same size as the output (filtered image)
                %I know this isn't the most brilliant code you've seen
                %but...
                if l == 1
                    b{f} = rand(24,24)*2-1;
                elseif l == 3
                    b{f} = rand(8,8)*2-1;
                end
            end
        elseif strcmp(layerTypes{l},'subsampling')
            for f = 1:numFilters(l)
                W{f} = ones(2,2)/4;
                b{f} = 0;
            end
        end
        Ws{l} = W;
        bs{l} = b;
        clear W b;
    end

    %weights and biases for the fully connected layer:
    Wfc = rand(10,192)*2-1;
    bfc = rand(10,1)*2-1;
else
    %change this to whatever you named your network of course
    load('LeNet3.mat');
end

seq = randperm(60000);  %generate random permutation of examples
errors = 0;
%This program does one epoch at a time, since the epochs are pretty long
%Each epoch takes about 3 or 4 hours on my laptop, using in i5 CPU with 4
%GB of RAM.

for iter = 1:length(seq)
    I = pixel(:,:,seq(iter));
    correct_output = zeros(10,1); correct_output(label(seq(iter))+1) = 1;
    
    %first convolutional layer:
    for f = 1:numFilters(1)
        H{1}{f} = conv_filter(I,Ws{1}{f},bs{1}{f});
    end

    %first subsampling layer:
    for f = 1:numFilters(2)
        H{2}{f} = subsampling_filter(H{1}{f},Ws{2}{f});
    end
    %second convolutional layer:
    %From my understanding, certain images are summed together before being
    %passed to the second convolutional layer. The summation is done
    %according to Table 1 of the paper.
    %This is the fifth layer I mentioned earlier.
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
    for f = 1:numFilters(4)
        H{5}{f} = subsampling_filter(H{4}{f},Ws{4}{f});
    end
    
    %Now we must prepare the input for the fully connected layer. This
    %basically takes each 4x4 subsampled image, transformes it into a 16x1
    %vector, and concatenates all of the 16x1 vectors to constitute one
    %giant 192x1 input vector for the fully connected layer.
    n = 1;
    for f = 1:size(H{5},2)
        X(n:n+numel(H{5}{f})-1,1) = reshape(H{5}{f},16,1);
        n=n+numel(H{5}{f});
    end
    %We'll have to do the opposite during back-propagation of course
    
    %Finally, here's the output to the network:
    final_output = logsig(Wfc*X+bfc);
    [whatever,given_answer] = max(final_output);
    %don't forget that 0 corresponds to the first neuron
    given_answer=given_answer-1;
    %disp(['correct answer: ' num2str(label(iter))]);
    %disp(['given answer: ' num2str(given_answer-1)])
    if (given_answer ~= label(seq(iter)))
        errors = errors+1;
    end

    MSE = sum((final_output-correct_output).^2)/10;

    O = final_output;
    T = correct_output;
    
    %comment this for faster performance (not by much but still)
    disp(['iter: ' num2str(iter) ' MSE = ' num2str(MSE)])

    %--------------------------------------------------------
    %Backpropagation time:
    %dE_dbfc means dE/dbfc
    dE_dbfc = 2*(O-T).*O.*(1-O);
    dE_dWfc = dE_dbfc*X';

    %we need dE/dX to backpropagate further:
    dE_dX = Wfc'*dE_dbfc;

    %remember X is a giant 192x1 vector containing all concatenated 16x1 
    %images, and so is dE/dX. We need to do some cleaning before we can
    %move on. The cleaning consists of separating the 12 16x1 subvectors in 
    %dE/dX. We'll use a temporary variable named dE/dX2
    dE_dX2 = cell(1,12);
    for i=1:12
        dE_dX2{i} = dE_dX((i-1)*16+1:i*16);
    end
    dE_dX = dE_dX2;clear dE_dX2;
    
    %Backpropagate through the second subsampling layer:
    %dE/dH5 is just dE/dX but with the values rearranged into matrices

    dE_dH4 = cell(1,12);
    dE_dH5 = cell(1,12);
    for f=1:12
        dE_dH5{f} = reshape(dE_dX{f},4,4);
        dE_dH4{f} = zeros(8,8);
        for i=1:4
            for j=1:4
                dE_dH4{f}(i*2-1:i*2,j*2-1:j*2) = dE_dH5{f}(i,j)*0.25;
            end
        end
    end

    %Compute dH4/dW3 and dH4/dB3
    dH4_dW3 = cell(1,12);
    dH4_dB3 = cell(1,12);
    for f = 1:12
        [dH4_dW3{f},dH4_dB3{f}] = conv_layer_grad(H{3}{f},H{4}{f},w);
    end

    %Now use dH4/dW3 and dH4/dB3 to compute dE/dW3 and dE/dB3
    dE_dW3 = cell(1,12);
    dE_dB3 = cell(1,12);
    for f=1:12
        dE_dW3{f} = zeros(w);
        for i=1:8
            for j=1:8
                dE_dW3{f} = dE_dW3{f} + dE_dH4{f}(i,j).*dH4_dW3{f}{i,j};
            end
        end
        dE_dB3{f} = dE_dH4{f}.*dH4_dB3{f};
    end

    %We also need dH4/dH3 in order to backpropagate to the first
    %convolutional layer
    dH4_dH3 = cell(1,12);
    for f=1:12
        dH4_dH3{f} = conv_layer_grad_wrt_inputs(H{3}{f},H{4}{f},Ws{3}{f});
    end
    %dE/dH3 = dE/dH4 * dH4/dH3:
    dE_dH3 = cell(1,12);
    for f=1:12
        dE_dH3{f} = zeros(size(H{3}{f}));
        for i=1:8
            for j=1:8
                dE_dH3{f} = dE_dH3{f} + dE_dH4{f}(i,j).*dH4_dH3{f}{i,j};
            end
        end
    end
    
    %We assumed that H3 was the result of the summations of pairs of the
    %images in H2. A little calculus should help you see that dH3/dH2 is
    %nothing but table 1 all over again.
    dH3_dH2 = [1,0,1,1,1,1,0,0,0,0,0,0;
               0,1,1,1,1,1,0,0,0,0,0,0;
               0,0,0,0,0,0,1,0,1,1,1,1;
               0,0,0,0,0,0,0,1,1,1,1,1];
    dE_dH2 = cell(1,4);
    %again, dE/dH2 = dE/dH3 * dH3/dH2.
    for f1=1:4
        dE_dH2{f1} = 0;
        for f2=1:12
            dE_dH2{f1} = dE_dH2{f1} + dE_dH3{f2}*dH3_dH2(f1,f2);
        end
    end
    
    %Again, backpropagate through a subsampling layer, this was already
    %done once
    dE_dH1 = cell(1,4);
    for f=1:4
        dE_dH1{f} = zeros(8,8);
        for i=1:12
            for j=1:12
                dE_dH1{f}(i*2-1:i*2,j*2-1:j*2) = dE_dH2{f}(i,j)*0.25;
            end
        end
    end
    
    %Finally, finde dH1/dW1 and dH1/dB1 in order to compute dE/dW1 and
    %dE/dB1, and we're done.
    dH1_dW1 = cell(1,4);
    dH1_dB1 = cell(1,4);
    for f = 1:4
        [dH1_dW1{f},dH1_dB1{f}] = conv_layer_grad(I,H{1}{f},w);
    end

    dE_dW1 = cell(1,4);
    dE_dB1 = cell(1,4);
    for f=1:4
        dE_dW1{f} = zeros(w);
        for i=1:24
            for j=1:24
                dE_dW1{f} = dE_dW1{f} + dE_dH1{f}(i,j).*dH1_dW1{f}{i,j};
            end
        end
        dE_dB1{f} = dE_dH1{f}.*dH1_dB1{f};
    end
    
    %We're done computing the necessary gradients. Weights update time:
    for f=1:4
        Ws{1}{f} = Ws{1}{f} - 0.1*dE_dW1{f};
        bs{1}{f} = bs{1}{f} - 0.1*dE_dB1{f};
    end

    for f=1:12
        Ws{3}{f} = Ws{3}{f} - 0.1*dE_dW3{f};
        bs{3}{f} = bs{3}{f} - 0.1*dE_dB3{f};
    end

    Wfc = Wfc - 0.1*dE_dWfc;
    bfc = bfc - 0.1*dE_dbfc;
end

disp('Done.')
disp(['Number of errors: ' num2str(errors)]);
%save the network
save('LeNet.mat','Ws','bs','Wfc','bfc');
%This next script is called automatically to test the trained network
%You should have good results even after only 1 epoch, since the number of
%training images is so large.
LeNet_test