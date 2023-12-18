import math
from .semiautograd import Function

Pow   = Function("Pow",   lambda x,p: x**p,        lambda x, p: [p*(x**(p-1))])

Plus  = Function("Plus",  lambda x,y: x+y,         lambda x,y: [1,1])

Sum   = Function("Sum",   lambda *args: sum(args), lambda *args: [1]*len(args))

Times = Function("Times", lambda x,y: x*y,         lambda x,y: [y, x])

Mod   = Function("Mod",   lambda x,m: x % m,       lambda x, m: [1])

Abs   = Function("Abs",   lambda x: abs(x),        lambda x: [-1 if x<0 else 1])

Cos   = Function("Cos",   lambda x: math.cos(x),   lambda x: [-math.sin(x)])

Sin   = Function("Sin",   lambda x: math.sin(x),   lambda x: [math.cos(x)])

Exp   = Function("Exp",   lambda x: math.exp(x),   lambda x: [math.exp(x)])

Log   = Function("Log",   lambda x: math.log(x),   lambda x: [1/x])