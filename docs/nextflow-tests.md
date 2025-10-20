# Nextflow Tests

[![nf-test](https://img.shields.io/badge/tested_with-nf--test-337ab7.svg)](https://code.askimed.com/nf-test)

We test the nextflow pipeline using `nf-test`. The docs for configuring and running tests are located at https://www.nf-test.com/.

## Running Tests

To detect all available tests:

`nf-test list tests/`

To run all tests in the tests folder:

`nf-test test tests/ -c nf-test.config`

### Running Integration tests on hpc:

Setup:
```
module use --append /projects/kumar-lab/meta/modules
module load nextflow
module load nf-test
```

Running the tests:

`nf-test test tests/ --profile sumner2 --tag feature,tracking --verbose`

## Tags

Tags can be used to select a subset of tests.

* `integration`: Test validates multiple components working together. This can include depending on external data or servers.
* `unit`: Test validates a single independent component.
* `tracking`: Test validates a component of pose file generation (python code in this repository).
* `feature`: Test validates a component of feature generation.
* `jabs`: Test involves usage of JABS-behavior-classifier.
* `remote`: Test involves remote servers (depends on machines outside nextflow's control).
* `globus`: Test requires globus configuration to be active (globus endpoints with active permissions).
* `rclone`: Test requires rclone configuration to be active (active token for remote server).
