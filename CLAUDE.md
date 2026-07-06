1. This is a project that lets me study intra-memory knowledge conflict and develop efficient approaches to
mitigate its damage. As definition, intra-memory conflict means that both plausible versions of a question
are within the model's parametric knowledge, as opposed to context-memory conflict. An example of intra-
memory conflict would be who the president is, as model learned many answers from news at different times.
Tentative research question: how does frequency and recency causally determine which of two competing
memorized facts wins, and how can we effectively intervene on that outcome, at train time or run time?

2. The runs will be done remotely on a GPU cluster. For basic runs I will likely allocate a single
A40 GPU, but we could upgrade to multiple GPU or A100 (80gb) if necessary. The maximum wall time allowed by
the cluster policy is 12 hours. I included a sample_sbatch.sbatch for sample header file, and the slurm 
standard out should go into the slurm/ folder. It is advisable to ask for two times the time you anticipate
the job would need, provided it is not over 12 hours. Use the epis conda environment for everything.

3. There are three main stages that I think are needed. These are a very rough outline, and when ask you to
run or build something, I will clearly indicate which phase we are on. 
a) We will construct a synthetic dataset that will provide the basic training corpus. We will carefully 
manage when the conflicting data will occur, how many times it occur, and in what forms it occur.
b) We will train a model (default GPT2 small from scratch on constructed corpus) We will test many baselines
and evaluate their performance. There are three main classes of baselines. i) Prompting. We will test many 
classical knowledge conflict prompts and evaluate performance. ii) Inference-time interference, such as
steering, patching, ablation, SAE based methods. iii) Training time interference, such as deduplication,
reweightings, etc.
c) We will either come up with a novel technique to better mitigate these problems, or we will use mechanistic
interpretation techniques that will determine where these happen, either during inference time or train time.
This is just the rough guideline. Specifically, we will create an experimental_plans.tex file that acts as 
the methods section of a paper. We will use that to make detailed decisions on the experiment and follow them
in implementation. The experimental_plans.tex should be very concise, roughly divided to the three sections
we discussed above, but contain most decision points and implementation. Clear citations are also needed, and
you will create references.bib to keep track of the files you cited. You may NOT render the pdf at all times.

4. There needs to be specialized folder for raw input data, output (such as weights), results (which are more
organized than output, such as json), and for plots. Write a README.md to document the file structure. There
need to be folders for each of training, preprocess, prompting-based techniques, inference-time interference,
training-time interference, and mech interp tools. We will try different methods within each module, and they
deserve different scripts, unless for shared methods. There also need to be an experiments/ folder that records
separately each experiment we run. hen naming, do not use non-discriptive numbers such as experiment 1 or exp_3.
Use discriptive names, such as describe it uses activation steering at inference time. 

5. As a general guideline, write clean and modular code. There should be strong logging and restart mechanism if the 
duration of the run is expected over an hour. As general rule, a checkpoint should be about every 10 minutes. When I
asked you to run an experiment, always checkin with me about the intended design before you submit the actual jobs.
You always need to confirm with me the estimated experiment duration and resources needed, for instance, one A100
for 3 hours. While testing the code is good practice, when I am not using remote server (i.e., no GPU present,
check by nvidia-smi), then you may NOT run any test that involve loading/running a language model.

6. I use git to manage the system and for syncing work. You should not commit directly unless I give explicit
permission. You should make the pipeline and file name such that only the most important things will get picked
up by git (i.e., not in .gitignore). For instance, I do not want the detailed log and .pt files to be saved, but
I want the final json to be backed up.
