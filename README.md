# code_sample
Code sample for the application to a position as a postdoctoral research fellow at the Institute of Theoretical Astrophysics in Oslo.

## Overview
This repository contains a code sample demonstrating a High-Performance Computing (HPC) pipeline designed to fit gNFW pressure profiles to galaxy cluster maps using multi-frequency data from **Planck** and **SPT**. 

This software performs a joint fit of the 'universal profile' across a set of galaxy clusters. It represents an evolution of the methods for single clusters described in *Oppizzi et al., 2023 (arXiv:2209.09601)*.

It is engineered for a specific High Performance Computing (HPC) architecture (distributed physical nodes, each featuring 40 shared-memory CPUs) and leverages a hybrid MPI/Ray parallelization strategy to maximize efficiency.

**Note:** This script is provided as a static code sample for evaluation purposes. Dependencies such as internal utility libraries and proprietary datasets are not included, rendering the script non-executable in this standalone form.
