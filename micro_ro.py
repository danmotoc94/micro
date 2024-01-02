import math
from graphviz import Digraph
import random
from typing import Any
import numpy as np

class Valoare:
  
    def __init__(self, data, _copii=(), _op='', eticheta=''):
        # Constructorul clasei Valoare.
        # data: Valoarea reală a obiectului (valoarea propriu-zisă).
        # _copii: O listă de noduri copii ale obiectului în cadrul graficului de calcul.
        # _op: Eticheta operației efectuate asupra acestui nod.
        # eticheta: O etichetă opțională pentru a identifica variabila (folosită în scopuri de debug sau documentare).
        self.data = data
        self.grad = 0.0  # Gradientul inițializat la 0.0
        self._inapoi = lambda: None  # Funcție de backward inițializată la o lambda care nu face nimic.
        self._precedent = set(_copii)  # Setul de noduri copii ale acestui nod.
        self._op = _op  # Eticheta operației efectuate.
        self.eticheta = eticheta  # Eticheta opțională pentru variabilă.

    def __repr__(self):
        # Reprezentarea string a obiectului Valoare.
        return f"Valoare(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Valoare) else Valoare(other)
        out = Valoare(self.data + other.data, (self, other), '+')
        
        # Backward pentru adunare.
        def _inapoi():
            # Calculul gradientului pentru adunare.
            # d(out)/d(self) = 1, d(out)/d(other) = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._inapoi = _inapoi
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Valoare) else Valoare(other)
        out = Valoare(self.data * other.data, (self, other), '*')
        
        # Backward pentru înmulțire.
        def _inapoi():
            # Calculul gradientului pentru înmulțire.
            # d(out)/d(self) = other.data, d(out)/d(other) = self.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._inapoi = _inapoi
          
        return out
    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __radd__(self, other): # other + self
        return self + other
        
    
    def __pow__(self, other):
        out = Valoare(self.data**other, (self,), f'**{other}')

        # Backward pentru ridicare la putere.
        def _inapoi():
            # Calculul gradientului pentru ridicare la putere.
            # d(out)/d(self) = other * self.data^(other-1), d(out)/d(other) = self.data^other * ln(self.data)
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._inapoi = _inapoi

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Valoare(t, (self, ), 'tanh')
        
        # Backward pentru tangenta hiperbolică.
        def _inapoi():
            # Calculul gradientului pentru tangenta hiperbolică.
            # d(out)/d(self) = 1 - tanh^2(x)
            self.grad += (1 - t**2) * out.grad
        out._inapoi = _inapoi
        
        return out
    
    def exponențiala(self):
        x = self.data
        out = Valoare(math.exp(x), (self, ), 'exp')
        
        # Backward pentru funcția exponențială.
        def _inapoi():
            # Calculul gradientului pentru funcția exponențială.
            # d(out)/d(self) = exp(x)
            self.grad += out.data * out.grad 
        out._inapoi = _inapoi
        
        return out
    
    def inapoi(self):
        # Inițierea pasului de backward. Construirea unei liste topologice și apoi propagarea gradientelor înapoi.

        topo = []  # Lista topologică pentru a păstra ordinea corectă a nodurilor în backward.
        vizitate = set()  # Set pentru a ține evidența nodurilor vizitate.
        
        # Funcție auxiliară pentru construirea listei topologice.
        def construieste_topo(v):
            if v not in vizitate:
                vizitate.add(v)
                for copil in v._precedent:
                    construieste_topo(copil)
                topo.append(v)
        
        # Apelarea funcției pentru a construi lista topologică pornind de la nodul curent.
        construieste_topo(self)
        
        # Inițierea gradientului la 1.0 pentru nodul curent.
        self.grad = 1.0
        
        # Propagarea gradientelor înapoi, în ordinea inversă a listei topologice.
        for nod in reversed(topo):
            nod._inapoi()

def urmarestecalea(radacina):
    # Construiește un set de toate nodurile și muchiile într-un graf
    noduri, muchii = set(), set()

    def construieste(v):
        if v not in noduri:
            noduri.add(v)
            for copil in v._precedent:
                muchii.add((copil, v))
                construieste(copil)

    construieste(radacina)
    return noduri, muchii

def deseneazagraf(radacina):
    dot = Digraph(format='pdf', graph_attr={'rankdir': 'LR'})  # LR = de la stânga la dreapta

    noduri, muchii = urmarestecalea(radacina)
    for n in noduri:
        uid = str(id(n))
        # Pentru orice valoare în graf, creează un nod rectangular ('record') pentru ea
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.eticheta, n.data, n.grad), shape='rect')
        if n._op:
            # Dacă această valoare este rezultatul unei operații, creează un nod pentru operație
            dot.node(name=uid + n._op, label=n._op)
            # Și conectează acest nod la el
            dot.edge(uid + n._op, uid)

    for n1, n2 in muchii:
        # Conectează n1 la nodul operației n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

# Derivata funcției LOSS L în raport cu greutățile a, b, c, d, e, f
# cum afectează greutățile funcția LOSS
# propagare înapoi manuală
# import numpy as np
# import matplotlib.pyplot as plt
# plt.plot(np.arange(-5,5,0.2), np.tanh(np.arange(-5,5,0.2))); plt.grid();
# plt.show()

# Intrările x1, x2
x1 = Valoare(2.0, eticheta='x1')
x2 = Valoare(0.0, eticheta='x2')

# Greutățile w1, w2
w1 = Valoare(-3.0, eticheta='w1')
w2 = Valoare(1.0, eticheta='w2')

# Bias-ul neuronului
b = Valoare(6.8813735870195432, eticheta='b')

# x1*w1 + x2*w2 + b
x1w1 = x1 * w1
x1w1.eticheta = 'x1*w1'

x2w2 = x2 * w2
x2w2.eticheta = 'x2*w2'

x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.eticheta = 'x1*w1 + x2*w2'

n = x1w1x2w2 + b
n.eticheta = 'n'

o = n.tanh()
o.eticheta = 'o'
o.grad = 1.0

o.inapoi()

deseneazagraf(o).render(filename='grafica_calculului', format='pdf')
class Neuron:
    def __init__(self, nin):
       self.w = [Valoare(random.uniform(-1,1)) for _ in range(nin)]
       self.b = Valoare(random.uniform(-1,1))
    def __call__(self, x):
       act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
       out = act.tanh()
       return out
    def parameters(self):
       return self.w + [self.b]

class Layer:
   def __init__(self, nin, nout):
       self.neurons = [Neuron(nin) for _ in range(nout)]
   def __call__(self, x):
       outs = [n(x) for n in self.neurons]
       return outs[0] if len(outs) == 1 else outs
   def parameters(self):
       return [p for neuron in self.neurons for p in neuron.parameters()]
# Layer este numele clasei, care reprezintă un strat într-o rețea neurală.
# Metoda __init__ este constructorul clasei Layer. Este apelată atunci când se creează un obiect Layer nou.
# self este o referință la instanța clasei care este creată. În acest caz, se referă la obiectul Layer.
# nin este un parametru care reprezintă numărul de caracteristici sau neuroni de intrare care se conectează la acest strat.
# nout este un parametru care reprezintă numărul de neuroni din acest strat.
# self.neurons este o variabilă de instanță care stochează o listă de obiecte Neuron. 
# Fiecare obiect Neuron este creat cu nin conexiuni de intrare. 
# Comprehensiunea listei [Neuron(nin) for _ in range(nout)] creează nout obiecte Neuron și le stochează în lista self.neurons.
# În rezumat, clasa Layer este folosită pentru a crea un strat de neuroni pentru o rețea neurală. 
# Numărul de conexiuni de intrare (nin) și numărul de neuroni din strat (nout) sunt specificate atunci când se creează un obiect Layer nou. 
# Stratul constă în mai multe obiecte Neuron, fiecare având nin conexiuni de intrare.

class MLP:
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

x = [2.0, 3.0, 4.4]
n = MLP(3, [4, 4, 1])
print(*n.parameters(), sep='\n')
# draw_dot(n(x)).render(filename='computation_graph', format='svg')
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
ypred = [n(x) for x in xs]
print(ypred)
loss = sum([(Valoare(ygt)-yout)**2 for ygt, yout in zip(ys, ypred)])
print(loss)
loss.inapoi()
deseneazagraf(loss).render(filename='computation_graph', format='svg')

for k in range(20):
  
  # forward pass
  ypred = [n(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  
  # backward pass
  for p in n.parameters():
    p.grad = 0.0
  loss.inapoi()
  
  # update
  for p in n.parameters():
    p.data += -0.1 * p.grad
  
  print(k, loss.data)
