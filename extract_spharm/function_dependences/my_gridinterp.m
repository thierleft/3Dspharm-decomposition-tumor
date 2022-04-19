function F = my_gridinterp( IMG )
    
    [ydim, xdim, zdim] = size(IMG); 
    [X,Y,Z] = meshgrid( 1 : xdim,  1 : ydim,  1 : zdim );
    X = X - median(X(:)); Y = Y - median(Y(:)); Z = Z - median(Z(:)); 

    V = IMG;

    P = [2 1 3];
    X = permute(X, P);
    Y = permute(Y, P);
    Z = permute(Z, P);
    V = permute(V, P);

    F = griddedInterpolant(X, Y, Z, V, 'nearest', 'nearest');

end
