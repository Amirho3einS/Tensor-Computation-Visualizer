# Tensor Contraction Visualizer

An interactive browser app for exploring tensor contractions written as einsum expressions. It parses contractions such as `ik,kj->ij`, displays the input and output tensors, and animates the vectorized execution step by step.

The visualizer helps inspect:

* valid output dimensions for vectorization
* tensor lanes, scalar broadcasts, and output accumulation
* hardware-oriented tensor layouts and required transposes
* scalar loop nests alongside vectorized loop nests
* computed output values against the expected contraction result

## Run Locally

**Prerequisite:** Node.js

1. Install dependencies:

   ```bash
   npm install
   ```

2. Start the development server:

   ```bash
   npm run dev
   ```

3. Open the local URL printed by Vite.
