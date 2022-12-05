function [dOdW,dOdB] = conv_layer_grad(I,O,w)

%the function returns the gradient of W for every position of the filter
%position in I. the program calling this function will use each of these Ws
%gradients to calculate the total gradient of W. we can't compute it here
%because we'll need to multiply each gradient by the input value from the
%input image of the convolutional layer


for i = 1:size(O,1)
    for j = 1:size(O,2)
        dOdW{i,j} =  O(i,j).*(1-O(i,j)).*I(i:i+w-1,j:j+w-1);
    end
end

dOdW = dOdW;
dOdB = O.*(1-O);