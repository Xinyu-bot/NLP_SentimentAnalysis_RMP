Full commands flow:
* At main: `git pull`
* __Create__ a new branch if needed: `git branch new_branch_name`
* __Switch__ to the new branch: `git checkout new_branch_name`
  * Create and Switch can be done in one composite command: `git checkout -b new_branch_name`
* __Start working on new changes...__
* For each changes:
  * __stage__ the change `git add changed_file_name`, please restrict yourself from using `git add --all` although `git add some_dir/.` is Okay. 
  * __commit__ the change `git commit -m "message"`
  * try to separate the adds and commits instead of doing a big add and commit including everything, because it will ruin the version control. 
* Before actually pushing, __pull from remote__ to update local files one last time in case someone else has already made change to the remote repository: `git pull origin main`. Fix any conflict locally (which is simple if using VS Code). 
* Finally, __push__ all to Github: `git push --set-upstream origin new_branch_name`
* Then in the GitHub website, find the branch and check if the branch has commit(s) behind or not. If the branch is not behind (guaranteed by following the previous steps), make a __pull request__ 
* ask for a review from others, or self-review. If no obvious bug or error, __merge__ into main branch.