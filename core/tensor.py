import numpy as np

class Tensor(object):
    def __init__(self, data, creation_op=None, creators=None, autograd=False, id=None):
        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None
        self.autograd = autograd
        self.children = {}
        if(id is None):
            id = np.random.randint(0,100000)
        self.id = id

        if creators is not None:
            for creator in creators:
                if(self.id not in creator.children):
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1

    def fully_backpropagated(self):
        for id,cnt in self.children.items():
            if(cnt != 0):
                return False
        return True

    
    def __repr__(self):
        return str(self.data.__repr__())

    """Tensor Operations"""

    def __add__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data + other.data, creation_op="add", creators=[self, other], autograd=True)
        return Tensor(self.data+other.data)

    def __neg__(self):
        if(self.autograd):
            return Tensor(self.data*-1,  creation_op="neg", creators=[self], autograd=True)
        return Tensor(self.data*-1)

    def __sub__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data - other.data, creation_op="sub", creators=[self, other], autograd=True)
        return Tensor(self.data-other.data)
        
    def __mul__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data*other.data, creation_op="mul", creators=[self ,other], autograd=True)
        return Tensor(self.data*other.data)

    def sum(self, dim):
        if(self.autograd):
            return Tensor(self.data.sum(dim), creation_op="sum_"+str(dim), creators=[self], autograd=True)
        return Tensor(self.data.sum(dim))

    def expand(self, dim, n_copies):
        transpose_dim = list(range(0, len(self.data.shape))).insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [n_copies]
        new_data = self.data.repeat(n_copies).reshape(new_shape)
        new_data = new_data.transpose(transpose_dim)
        if(self.autograd):
            return Tensor(new_data, creation_op="expand_"+str(dim), creators=[self], autograd=True)
        return Tensor(new_data)

    def transpose(self):
        if(self.autograd):
            return Tensor(self.data.transpose(), creation_op="transpose", creators=[self], autograd=True)
        return Tensor(self.data.transpose())

    def mm(self, x):
        if(self.autograd):
            return Tensor(self.data.dot(x.data), creation_op="mm", creators=[self,x], autograd=True)
        return Tensor(self.data.dot(x.data))

    def sigmoid(self):
        if(self.autograd):
            return Tensor(1/(1+np.exp(-self.data)), creation_op="sigmoid", creators=[self], autograd=True)
        return Tensor(1/(1+np.exp(-self.data)))

    def tanh(self):
        if(self.autograd):
            return Tensor(np.tanh(self.data), creation_op="tanh", creators=[self], autograd=True)
        return Tensor(np.tanh(self))

    def index_select(self, indices):
        if(self.autograd):
            new = Tensor(self.data[indices.data], creation_op = "ind_select", creators = [self], autograd=True)
            new.indices_selected = indices
            return new
        return Tensor(self.data[indices.data])

    def cross_entropy(self, target_indices):
        temp = np.exp(self.data)
        softmax = temp/np.sum(temp, axis=len(self.data.shape)-1, keepdims=True)
        t = target_indices.data.flatten()
        p = softmax.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (target_dist)).sum(1).mean()
        if(self.autograd):
            out = Tensor(loss, creation_op="cross_entropy", creators=[self], autograd=True)
            out.target_dist = target_dist
            out.softmax = softmax
            return out
        return Tensor(loss)

    def __str__(self):
        return str(self.data.__str__())

    def backward(self, grad=None, source=None):
        if(self.autograd):
            if(source is not None):
                if self.children[source.id] == 0:
                    raise Exception("No more backprops from this tenaor left")
                else:
                    self.children[source.id] -= 1

            if(self.grad) == None:
                    self.grad = grad
            else:
                self.grad += grad
            if(self.creators is not None and (self.fully_backpropagated() or source is None)):
                if self.creation_op == "add":
                    for creator in self.creators:
                        creator.backward(self.grad, self)
                elif self.creation_op == "neg":
                    for creator in self.creators:
                        creator.backward(self.grad.__neg__())
                elif self.creation_op == "sub":
                    new = Tensor(self.grad.data)
                    self.creators[0].backward(new, self)
                    new = Tensor(self.grad.__neg__().data)
                    self.creators[1].backward(new, self)
                elif self.creation_op == "mul":
                    new = self.grad* self.creators[1]
                    self.creators[0].backward(new, self)
                    new = self.grad*self.creators[0]
                    self.creators[1].backward(new, self)
                elif self.creation_op == "mm":
                    act = self.creators[0]
                    weights = self.creators[1]
                    new = self.grad.mm(weights.transpose())
                    act.backward(new)
                    new = self.grad.transpose().mm(act).transpose()
                    weights.backward(new)
                elif("sum" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    n_copies = self.creators[0].data.shape[dim]
                    self.creators[0].backward(self.grad.expand(dim, n_copies))
                elif("expand" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))
                elif self.creation_op == "transpose":
                    self.creators[0].backward(self.grad.transpose())
                elif self.creation_op == "sigmoid":
                    ones = np.ones_like(self.grad.data)
                    self.creators[0].backward(self.grad *  self.data*(ones-self.data))
                elif self.creation_op == "tanh":
                    ones = np.ones_like(self.grad.data)
                    self.creators[0].backward(self.grad * (ones - self.data*self.data))
                elif self.creation_op == "ind_select":
                    new_grad = np.zeros_like(self.creators[0].data)
                    indices = self.indices_selected.data.flatten()
                    grad_ = self.grad.data.reshape(len(indices), -1)
                    for i in range(len(indices)):
                        new_grad[indices[i]] += grad_[i]
                    self.creators[0].backward(Tensor(new_grad))
                elif self.creation_op == "cross_entropy":
                    dx = self.softmax - self.target_dist
                    self.creators[0].backward(Tensor(dx))Commit on 2026-02-10T12:00:00
Commit on 2026-02-16T12:00:00
Commit on 2026-02-18T12:00:00
