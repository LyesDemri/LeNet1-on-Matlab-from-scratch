function out = subsampling_filter(I,W)

out = zeros(size(I,1)/2,size(I,2)/2);
for i = 1:2:size(I,1)
    for j = 1:2:size(I,2)
        out(ceil(i/2),ceil(j/2)) = sum(sum(I(i:i+1,j:j+1).*W));
    end
end