function dOdI = conv_layer_grad_wrt_inputs(I,O,W)

w = length(W);
dOdI = cell(size(O));
for i = 1:size(O,1)
    for j = 1:size(O,2)
        dOdI{i,j} = zeros(size(I));
        dOdI{i,j}(i:i+w-1,j:j+w-1) = O(i,j)*(1-O(i,j)).*W;
    end
end
