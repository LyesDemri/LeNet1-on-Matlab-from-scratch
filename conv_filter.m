function out = conv_filter(I,W,b)

%W must be a square matrix with even number of rows and columns
m = floor(size(W,1)/2);
out = zeros(size(I,1)-2*m,size(I,2)-2*m);
for i = m+1:size(I,1)-m
    for j = m+1:size(I,2)-m
        out(i-m,j-m) = logsig(sum(sum(I(i-2:i+2,j-2:j+2).*W))+b(i-m,j-m));
    end
end