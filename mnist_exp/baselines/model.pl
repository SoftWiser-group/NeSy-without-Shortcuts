nn(mnist_net,[X],U,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,U).
nn(mnist_net,[Z],V,[0,1,2,3,4,5,6,7,8,9]) :: rdigit(Z,V).
rotate(X,Z) :- rdigit(Z,0);rdigit(Z,1);rdigit(Z,2);rdigit(Z,3);rdigit(Z,4);rdigit(Z,5);rdigit(Z,6);rdigit(Z,7);rdigit(Z,8);digit(X,6).
combine(X,Z,Y) :- rotate(X,Z), digit(X,Y).