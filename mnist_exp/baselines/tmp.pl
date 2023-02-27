nn(mnist_net,[X],U,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,U).
nn(mnist_net,[Z],V,[0,1,2,3,4,5,6,7,8,9]) :: rdigit(Z,V).
rotate(X,Z) :- rdigit(Z,0). 
rotate(X,Z) :- rdigit(Z,1).
rotate(X,Z) :- rdigit(Z,2). 
rotate(X,Z) :- rdigit(Z,3). 
rotate(X,Z) :- rdigit(Z,4). 
rotate(X,Z) :- rdigit(Z,5). 
rotate(X,Z) :- rdigit(Z,6). 
rotate(X,Z) :- rdigit(Z,7).
rotate(X,Z) :- rdigit(Z,8).
rotate(X,Z) :- digit(X,6).
combine(X,Z,Y) :- rotate(X,Z), digit(X,Y).