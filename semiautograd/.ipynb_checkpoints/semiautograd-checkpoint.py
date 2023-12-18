from typing import Any, Callable, Dict, List, Optional, Union

class Scalar:
    count = 0
    def __init__(self, value: float, function: Optional["Function"]=None, parents: List["Scalar"]=(), kwargs: Dict[str,Any]=None):
        '''Represent a variable in a differentiable calculation

        Arguments:
            value -- numerical value of the variable
            function -- function used to compute the variable (default None)
            parents -- Scalar inputs to that function (default ())
            kwargs -- non-Scalar parameters to that function
        '''
        self.value = value               
        self.function = function         
        self.parents = list(parents)        # HIGHLIGHT: parents defines the computational graph
        self.kwargs = kwargs or dict()  
        self.grad = None                # place-holder for the derivative
        self.time = self.__class__.count              # convenient topological sorting
        self.__class__.count += 1

    def __lt__(self, other):
        return self.value < (other.value if isinstance(other,Scalar) else other)

    def __gt__(self, other):
        return self.value > (other.value if isinstance(other,Scalar) else other)

    def __str__(self):
        gradstr = ''
        if self.grad is not None:
            gradstr = f' <grad={self.grad}>'
        if self.function is None:
            return f'{self.value}{gradstr}'
        argstr = ','.join([f'{n.value}' for n in self.parents] + [f"{k}={v}" for k,v in self.kwargs.items()])
        return f'{self.value} = {self.function.name}(' + argstr + ')'+gradstr

    def __repr__(self):
        return str(self)

    def add_grad(self, g: float):
        '''Accumulate some gradient at this Scalar'''
        if self.grad is None:
            self.grad=0
        self.grad += g

class Function:
    def __init__(self, name: str, forward: Callable, backward: Callable):
        '''Represents a differentiable calculation on Scalars

        Arguments:
            name -- just for printing
            forward -- compute the forward pass with signature (*floats, **kwargs) -> float         
            backward -- compute the derivative w.r.t. each float with signature (*floats, **kwargs) -> List[float]
        '''
    
        self.name = name            # just for printing
        self._forward = forward     # the
        self._backward = backward

    def __str__(self):
        return f'{self.name}()'

    def __repr__(self):
        return str(self)
        
    def __call__(self, *scalars, **kwargs) -> Union[Scalar, float]:
        ''' Call the forward method

        Arguments:
            scalars -- List[Scalar] or List[float], the inputs to the function
            kwargs -- the parameters to the function
        Returns:
            The output of the forward computation with the same type as in the scalars list
        '''
        isscalar = [isinstance(s,Scalar) for s in scalars]
        assert all(isscalar) or not any(isscalar), "wtf you trying to do?"
        if isinstance(scalars[0],Scalar):    
            return Scalar(
                value = self._forward(*[s.value for s in scalars], **kwargs),
                function = self,
                parents = scalars,
                kwargs=kwargs
            )
        return self._forward(*scalars, **kwargs)


    def d(self, x: Scalar) -> None:
        ''' Call the backward method and accumulate gradients on the parents of x

        Arguments:
            x -- The child node with its .grad calculation complete
        '''
        ds = self._backward(*[p.value for p in x.parents], **x.kwargs)
        for s,d in zip(x.parents,ds):
            s.add_grad(x.grad * d)     # HIGHLIGHT: Applying the chain rule!

def trace(x: Scalar) -> List[Scalar]:
    '''Topologically sort the computation graph to x'''
    nodes = []
    unexplored = [x]
    while len(unexplored)>0:  # DFS the computational graph
        n = unexplored.pop()
        nodes.append(n)
        for p in n.parents:
            if not any(p is o for o in nodes+unexplored):
                unexplored.append(p)
    return sorted(nodes, key=lambda n: n.time, reverse=True) # this is a topological sort b/c evaluation is eager with no parallelism

def backward(x: Scalar):
    '''Compute the derivative of x w.r.t. each Scalar in its graph'''
    x.grad = 1
    for n in trace(x):
        if n.function is not None:
            n.function.d(n)

def reset_grad(x: Scalar):
    '''Reset all the grads in trace(x) to None'''
    for n in trace(x):
        n.grad = None