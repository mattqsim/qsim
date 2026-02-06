## qsim

Minimal circuit simulator (statevector) with clean separation between 

- **Circuit** - description only (ops + metadata) 
- **Simulator** - executes a circuit and produces a statevector 
- **Apply** - performs k-qubit gate aplication via tensor reshaping 
- **Measurement** - explicit born sampling + collapse 
- **Evoultion** - provides time evolution helpers
- **Noise** - applied at execution time (stochastic Pauli-style noise hooks)

This repo aims to be a clean foundation for future work (Including QEC - quantum error correction)


- **Circuit ≠ execution**
  - `Circuit` objects describe *what* should happen.
  - Execution happens only inside the simulator.

- **Explicit physics**
  - No hidden state mutation
  - No backend flags
  - Measurement and collapse are explicit

- **Minimal abstractions**
  - Small functions
  - Clear data flow
  - Easy to reason about and test

## Quickstart Example

```python
from qsim.circuit import Circuit
from qsim.simulator import run_statevector

c = Circuit(2)
c.h(0)
c.cx(0, 1)

psi = run_statevector(c)
print(psi)
