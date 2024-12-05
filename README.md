# hartree_fock
>My attempt on coding Hartree Fock method

## Block diagonalize:
>	Takes a list of matrices and arranges them in a block diagonal form

$$
\left(\begin{array}{cc}
A && B && C && D
\end{array}\right)=
\left(\begin{array}{cc}
A && 0 && 0 && 0 \\
0 && B && 0 && 0\\
0 && 0 && C && 0\\
0 && 0 && 0 && D
\end{array}\right)
$$

> where A, B, C, D---- also are matrices

