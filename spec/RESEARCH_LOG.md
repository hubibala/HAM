# RESEARCH_LOG.md: The Holonomic Association Model (HAM)

**Project:** Holonomic Association Model (HAM) / HAMTools
**Date:** December 2, 2025
**Status:** Strategic Pivot to Core Infrastructure (v2.0)

## 1. Executive Summary

We are pivoting from building a specific "World Model" application to developing **`HAMTools`**, a general-purpose, differentiable library for Finsler Geometry.

Our initial experiments (v0.x) successfully demonstrated that Neural Finsler Metrics can learn asymmetric control dynamics (e.g., gravity, wind) that Riemannian models cannot. However, the lack of a rigorous, verified geometric backend makes scaling dangerous. We are pausing the "Moonshot" (AGI World Models) to build the "Rocket Engine" (a reliable, JAX-based geometry library).

---

## 2. The Research Chronology (Pivots)

This project has evolved through four distinct theoretical phases. This narrative forms the basis for the eventual paper's introduction.

### Phase 1: Semantic Geometry (The NLP Era)
* **Hypothesis:** Semantic relationships (A $\to$ B) are asymmetric. Standard embeddings (Dot Product) are symmetric ($A \cdot B = B \cdot A$).
* **Proposal:** Use a **Randers Metric** ($F = \sqrt{g} + \beta$) where $\beta$ represents "Contextual Wind."
* [cite_start]**Outcome:** Theoretically sound, but discrete word tokens provided a noisy, sparse signal for validating continuous differential geometry [cite: 116-117].

### Phase 2: The Physical Pivot (The Robotics Era)
* **Shift:** We moved from "Semantic Similarity" to "Physical Energy."
* **Insight:** "Cost" is equivalent to "Time" or "Work." The axioms of Finsler geometry map perfectly to control theory.
* **Validation:** The "Coriolis Cargo" experiment.
    * *Task:* Robot pushing cargo on a spinning surface.
    * [cite_start]*Result:* The model learned a "Westward Wind" (Coriolis force) purely from observation, achieving **92.4% Cosine Similarity** in zero-shot transfer [cite: 147-150].

### Phase 3: The Architecture Split (Navigator & Pilot)
* **Problem:** End-to-end learning entangled "Geometry" (Planning) with "Control" (Muscle movement).
* **Solution:** Decoupled the architecture.
    * **Navigator (HAM):** Lives in latent space ($S^2$). Plans the vector field.
    * **Pilot (RL):** Lives in tangent space. [cite_start]Executes the vector [cite: 126-127].

### Phase 4: The Rigor Pivot (Current State)
* **Problem:** Our solvers (`AVBD`) and metrics (`RandersFactory`) are entangled with the experimental code. We lack analytical ground truths (e.g., exact Zermelo solutions) to verify if the solver is accurate to $10^{-6}$ or just "learning to cheat."
* **Solution:** Extract the geometry engine into a standalone library (`HAMTools`).
* **Key Technical Change:** Adoption of the **Berwald Connection** over the Chern Connection to correctly model parallel transport of dynamical systems.

---

## 3. Current Technical Status (v0.1)

### 3.1. Validated Components
* [cite_start]**Randers Factory:** The parameterization $\beta = \tanh(W) \cdot \text{convexity\_factor}$ successfully prevents metric collapse [cite: 33-34].
* [cite_start]**AVBD Solver:** The Augmented Lagrangian approach converges reliably on hard constraints (Manifold + Physics) without needing projection operators [cite: 40-43].
* [cite_start]**Zermelo Convexity:** We have mathematically proven that ensuring $\|W\|_h < 1$ is sufficient for the Legendre transform to exist [cite: 21-22].

### 3.2. Known Deficiencies (To Be Fixed in v1.0)
* **Implicit Geometry:** Spray coefficients and curvatures are computed implicitly or ad-hoc. They are not exposed as API calls.
* **Mesh Incompatibility:** The current implementation assumes continuous manifolds (Spheres/Planes). It cannot handle discrete meshes (Stanford Bunny).
* **Homogeneity Violations:** The neural networks currently do not strictly enforce $F(x, \lambda v) = \lambda F(x, v)$, leading to potential drift in the Berwald connection.

---

## 4. The Research Plan (1 Year)

### Phase 1: The "Finsler-JAX" Library (Months 1-3)
**Goal:** A pip-installable library that rivals `geomstats`.
* **Objective:** Implement `FinslerMetric`, `Spray`, and `BerwaldTransport` as generic JAX operators.
* **Validation:** Reproduce the **Zermelo Navigation** analytical solution on a flat plane.
* **Deliverable:** `HAMTools` v0.1 (Continuous only).

### Phase 2: Discrete Geometry (Months 4-6)
**Goal:** Run Finsler physics on triangular meshes.
* **Objective:** Adapt `AVBDSolver` to work on graph edges.
* **Validation:** Compare "Neural Geodesics" against Dijkstra/Heat Method on dense graphs.
* **Deliverable:** Mesh support in `HAMTools`.

### Phase 3: Deep Finsler Learning (Months 7-9)
**Goal:** Learn the geometry from data.
* **Objective:** Train a `LearnedFinsler` metric (Neural Field) to reconstruct the metric of a known surface from observed trajectories.
* **Deliverable:** "Inverse Reinforcement Learning" demo using Finsler geometry.

### Phase 4: Publication (Months 10-12)
**Goal:** "HAMTools: A Differentiable Library for Asymmetric Manifold Learning."
* **Focus:** The library as the contribution, with the "World Model" as the flagship application.

---

## 5. Artifacts & References

* **Math Spec:** `MATH_SPEC.md` (Version 1.1, Berwald Edition)
* **Architecture:** `ARCH_SPEC.md` (Version 1.0)
* **Legacy Code:** `src/ham/` (To be refactored)
* **Validation Data:** `notebooks/geometry-check.ipynb` (To be expanded)