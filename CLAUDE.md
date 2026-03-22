Whenever adding a new package, update the requirements.txt and environment.yml files accordingly

Ensure that your code is modularized, so it remains easy to read and maintain.
Always add comments with mathematical formulae to the sections of the code that implement them
Ensure that your code relies on the best performance practices of PyTorch
DO NOT use short names for the variables. Prefer proper names: attn->attention; ch->channel; conv->convolution; etc.
You can assume that there is a created environment on the device, based on the ./particle_transformer_lowpt_tau/environment.yml file
Usage of 'eval' is unsafe, and STRICTLY FORBIDDEN

When debugging poor performance of the model, always look for a file of the training logs, and the checkpoint of the best results within the run. They will help you confirm or deny any findings. Of such files do not exist, ask the user to provide them.

When running the models locally, use MPS

Keep README.md up to date whenever adding new scripts, models, or changing the project structure

Read the instructions from CLAUDE.md to update your context when you start processing EVERY new message

## Testing

Follow TDD (Test-Driven Development) for ALL code changes — not just large ones:
1. Write tests FIRST that define expected behavior (they should fail — red phase)
2. Implement the code until all tests pass (green phase)
3. Refactor if needed while keeping tests green

Before submitting ANY changes, run the full test suite (`python -m pytest tests/` from `part/`) and confirm zero failures. Do not present changes as done until all tests pass.