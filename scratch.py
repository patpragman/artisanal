from sympy import *

n, p, f, s = symbols("n p f s")


def C(n, p, f, s):
    return (n + 2*p - f)/s + 1

def P(size):
    return size/2

e = simplify(C(C(C(128, p, f, s), p, f, s), p, f, s))

f_sub = 4
s_sub = 2

layer_one_output = simplify(C(64, p, f, s)).subs(p, 1).subs(f, f_sub).subs(s, s_sub)
pool_1 = simplify(P(layer_one_output))
layer_two_output = simplify(C(pool_1, p, f, s)).subs(p, 1).subs(f, f_sub).subs(s, s_sub)
pool_2 = simplify(P(layer_two_output))
layer_three_output = simplify(C(pool_2, p, f, s)).subs(p, 1).subs(f, f_sub).subs(s, s_sub)


print(layer_one_output)
print(layer_two_output)
print(layer_three_output)
