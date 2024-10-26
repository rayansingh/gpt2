# ECE408/CS483 Final Project - GPT-2

## Milestone 1

## Project Requirements

For this milestone, you will need to come up with a basic parallel implementation of all the forward pass kernels in the `kernels` folder.

## Implementation and Testing

To build your implemented kernels, run the command

    make

For this milestone, we provide a test script `output_verification.cu` for you to verify the correctness of your forward pass kernels. The script tests the functionality of each kernel you wrote by comparing their outputs to the provided CPU kernels. Note that the tests are run independently, which means the test result of each kernel only depends on the correctness of that kernel.

You can run the test script by uncommenting the line `srun ./output_verification` in `job.slurm` file and then running.

    sbatch job.slurm

The outputs of the testing script will be saved in the `GPT_CU.out` file. Feel free to modify the testing script for your own debugging purposes.

When you try to test your code for the first time, some model weights need to be downloaded through a weights download script in the provided `job.slurm` file. You do not need to make any modifications to it.

To clean, run

    make clean

This will remove all the files generated during the compilation and execution process.

## Timeline
By November 1, you can expect to present your understanding to the new project, and at least two of the kernels are implemented correctly.

By November 8, at least two thirds of the kernels are implemented correctly.

By November 15, all kernels are implemented correctly.

### Optional Testing

You can also compile and run the actual forward pass of the GPT-2 model with the following steps:

1. Compile the project with the command

        make gpt2

2. Modify the `job.slurm` file to run the compiled executable by uncommenting the line `srun ./gpt2` and then run

        sbatch job.slurm

The output will be saved in the `GPT_CU.out` file.

## Final Notes

Please understand that this is the first semester we are running this project, and it is only being done at a very small scale. We appreciate your initiative on helping us beta-test this project.

While this project can seem overwhelming at times, we are here to help you succeed. This is why we have weekly meetings, as they are not only there to check your progress but also to help you with any issues/challenges you may encounter. When it comes to grading, we may also prioritize your group's consistent and good quality effort over the final outcome of your project.

We are excited to hear from you soon, good luck!
