---
# Page settings
layout: default
keywords:
comments: false

# Hero section
title: Git
description:

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content:
        url: '#'
    next:
        content:
        url: '#'
---

## What is version control?

The VCSs solve one of the most common problems of software development: the fear for changing the software. A VCS will help you to get rid of being scared about changing your code.

## What is Git?

Git is a distributed version control system (DVCS), also known as decentralized. This means that every developer has a full copy of the repository.

## Install and Config Git

### Installing on Ubuntu

```
sudo apt-get update
sudo apt-get install git
```

### Git Configuration

Git comes with a tool called git config that lets you get and set configuration variables that control all aspects of how Git looks and operates. These variables can be stored in three different places:

1. /etc/gitconfig file: Contains values for every user on the system and all their repositories. If you pass the option --system to git config, it reads and writes from this file specifically.

2. ~/.gitconfig or ~/.config/git/config file: Specific to your user. You can make Git read and write to this file specifically by passing the --global option.

3. config file in the Git directory (that is, .git/config) of whatever repository you’re currently using: Specific to that single repository.

Each level overrides values in the previous level, so values in .git/config trump those in /etc/gitconfig.

### Identity

```
git config --global user.name "John Doe"
git config --global user.email johndoe@example.com
```

you need to do this only once if you pass the --global option, because then Git will always use that information for anything you do on that system. If you want to override this with a different name or email address for specific projects, you can run the command without the --global option when you’re in that project.

### Checking Settings

```
git config --list
git config user.name
```

### Getting Help

```
git [verb] --help
```

## Git Usage

### Creating a repository

```
git init
```

### Checking status

```
git status
```

### Staging

Files are added to the index. The Git index is where the data that is going to be saved in the commit is stored temporarily, until the commit is done.

```
git add
```

### Commits

A commit is a full snapshot of the repository, that is saved in the database.

```
git commit -m "comment"
```

### Hash

Git identifies each commit uniquely using SHA1 hash function, based on the contents of the committed files. So, each commit is identified with a 40 character-long hexadecimal string, like the following.

```
de5aeb426b3773ee3f1f25a85f471750d127edfe
```

Take into account that the commit message, commit date, or any other variable rather than the committed files’ content (and size), are not included in the hash calculation.

### History

```
git log --pretty=oneline --graph --decorate
```

### Branch

```
git branch [branch-name]                                  # creating a branch
git branch -d [branch-name]                               # delete a branch
git push origin --delete [branch-name]                    # delete a remote branch
git branch -vv                                            # list branch
git branch -r                                             # list remote branch
git checkout [branch-name]                                # switching branchs
git branch --set-upstream-to=[remote-name]/[branch-name]  # track remote branch
```

### Tag

```
git tag                                   # listing tags
git tag -a [tag-name] -m "comment"        # creating an annotated tag
git show [tag-name]                       # show tag
git checkout -b [branch-name] [tag-name]  # checkout tags
git push origin [tag-name]                # push tag
```

### Undoing

#### Removing files

To remove a file from Git, you have to remove it from your staging area and then commit.

If you remove a modified file, you must force the removal with the -f option. This is a safety feature to prevent accidental removal of data that hasn’t yet been recorded in a snapshot and that can’t be recovered from Git.

```
git rm [file]
```

Another useful thing you may want to do is to keep the file in your working tree but remove it from your staging area. In other words, you may want to keep the file on your hard drive but not have Git track it anymore.

```
git rm --cached [file]
```

#### Moving files

```
git mv [file-from] [file-to]
```

#### Modifying the last commit

```
git add forgotten_file
git commit --amend -m "comment"
```

#### Discard changes in working directory

```
git checkout [file]
```

#### Unstaging a Staged File

```
git reset HEAD [file]
```

#### Deleting commits

That is, if you make a soft reset, the commit(s) will be removed, but the modifications saved in that/those commit(s) will remain; and a hard reset, won't leave change made in the commit(s). If no flag is specified, the reset will be done softly.

```
git reset [--hard|--soft] HEAD^   # First parent of the current branch
git reset [--hard|--soft] HEAD~n  
```

#### Rewriting History

```
git rebase -i -p HEAD~n
git rebase --interactive --preserve-merges HEAD~n
```

```
pick f7f3f6d log
pick 310154e log
pick a5f4a0d log

# Rebase 710f0f8..a5f4a0d onto 710f0f8
#
# Commands:
#  p, pick = use commit
#  r, reword = use commit, but edit the commit message
#  e, edit = use commit, but stop for amending
#  s, squash = use commit, but meld into previous commit
#  f, fixup = like "squash", but discard this commit's log message
#  x, exec = run command (the rest of the line) using shell
#
# These lines can be re-ordered; they are executed from top to bottom.
#
# If you remove a line here THAT COMMIT WILL BE LOST.
#
# However, if you remove everything, the rebase will be aborted.
#
# Note that empty commits are commented out
```

```
edit f7f3f6d log
pick 310154e log
pick a5f4a0d log
```

```
git rebase -i HEAD~n
Stopped at f7f3f6d... changed my name a bit
You can amend the commit now, with

       git commit --amend

Once you’re satisfied with your changes, run

       git rebase --continue
```

### Remote repositories

#### Add/Remove a remote

```
git remote add [remote-name] [repo-url] # origin as default remote-name
git remote remove [remote-name]
```

#### Showing Your Remotes

```
git remote -v
git remote show [remote-name]
```

#### Fetch: updating remote references

Updated the reference to remote’s branch, but the changes have not been applied in the repository

```
git fetch [remote-name] [branch-name]
```

What has Git internally with this? Well, now, a directory .git/refs/remotes has been created, and, inside it, another directory, origin (because that’s the name we have given to the remote). Here, Git creates a file for each branch exiting in the remote repository, with a reference to it. This reference is just the SHA1 id of the last commit of the given branch of the remote repository. This is used by Git to know if in remote repo are any changes that can be applied to the local repository.

Merge the remote branch, which has just been updated; with the local branch.

```
git merge [remote-name]/[branch-name] # no fast-forward merge
```

#### Pull: fetching and merging remotes at once

```
git pull [remote-name] [branch-name]
```

#### Push: writing changes into the remote

```
git push [remote-name] --all                   # Updates the remote with all the local branches
git push [remote-name] [branch-name]           # Updates remote’s branches
git push [remote-name] [tag-name]              # push tag
git push [remote-name] --tags                  # Sends tags to remotes
git push [remote-name] --delete [branch-name]  # delete remote branch
```

#### Clone a repository

```
git clone [repo-url]
```

By default, when cloning a repository, only the default branch is created (master, generally). The way to create the other branches locally is making a checkout to them.

### Merge

#### Avoid using fast-forward mode

When we are merging branches, is always advisable not to use the fast-forward mode. This is achieved passing --noff flag while merging, since the history is reflected as it is actually is. The no fast-forward mode should be always used.

```
git merge --no-ff branch-name
```

### Diff

#### Differences between working-directory and staged

```
git diff
```

#### Differences between staged and last-commited

```
git diff --cached
```

### Rebasing

**Do not rebase commits that exist outside your repository**

```
git checkout branch-name                        # switch to branch-name
git rebase target-branch-name                   # rebase onto target-branch-name
git checkout target-branch-name                 # switch to target-branch-name
git merge branch-name                           # do a fast-forward merge
git branch -d branch-name                       # delete branch-name
```

## Git Flow

### Installing on Ubuntu

```
sudo apt-get install git-flow
```

### Initialization

```
git flow init [-d]
```

### A successful Git branching model

![A successful Git branching model](/images/git/branching-model.png)

### Creating feature/release/hotfix/support branches

#### To list/start/finish feature branches

```
git flow feature
git flow feature start <name> [<base>]
git flow feature finish <name>
```

For feature branches, the [base] arg must be a commit on develop

#### To push/pull a feature branch to the remote repository

```
git flow feature publish <name>
git flow feature pull <remote> <name>
```

#### To list/start/finish release branches

```
git flow release
git flow release start <release> [<base>]
git flow release finish <release>
```

For release branches, the [base] arg must be a commit on develop

#### To list/start/finish hotfix branches

```
git flow hotfix
git flow hotfix start <release> [<base>]
git flow hotfix finish <release>
```

For hotfix branches, the [base] arg must be a commit on master

## Connecting to GitHub with SSH

### Checking for existing SSH keys


To see if existing SSH keys are present:

```
$ ls -al ~/.ssh
```

### Generating a new SSH key and adding it to the ssh-agent

This creates a new ssh key, using the provided email as a label.

```
$ ssh-keygen -t rsa -b 4096 -C "email@example.com"
```

Start the ssh-agent in the background.

```
$ eval "$(ssh-agent -s)"
Agent pid 59566
```

Add your SSH private key to the ssh-agent.

```
$ ssh-add ~/.ssh/id_rsa
```

### Adding a new SSH key to your GitHub account

* Copies the contents of the id_rsa.pub file.
* In the upper-right corner of any page, click your profile photo, then click **Settings**.
* In the user settings sidebar, click **SSH and GPG keys**.
* Click **New SSH key** or **Add SSH key**.
* In the "Title" field, add a descriptive label for the new key.
* Paste your key into the "Key" field.
* Click **Add SSH key**.
* If prompted, confirm your GitHub password.

### Testing your SSH connection

```
$ ssh -T git@github.com
# Attempts to ssh to GitHub
```

You may see a warning like this:

```
The authenticity of host 'github.com (IP ADDRESS)' can't be established.
RSA key fingerprint is 16:27:ac:a5:76:28:2d:36:63:1b:56:4d:eb:df:a6:48.
Are you sure you want to continue connecting (yes/no)?
```

or like this:

```
The authenticity of host 'github.com (IP ADDRESS)' can't be established.
RSA key fingerprint is SHA256:nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8.
Are you sure you want to continue connecting (yes/no)?
```

Verify that the fingerprint in the message you see matches one of the messages, then type **yes**:

```
Hi! You've successfully authenticated, but GitHub does not provide shell access.
```

## TroubleShooting

### Git Clone : The remote end hung up unexpectedly

```
git config --global http.postBuffer 524288000
```

## Resources

[Git Pro](https://git-scm.com/book/en/v2)

[Git Reference](https://git-scm.com/docs)

[Git Flow](https://github.com/nvie/gitflow)

[Connecting to GitHub with SSH](https://help.github.com/articles/connecting-to-github-with-ssh/)

[A successful git branching model](http://nvie.com/posts/a-successful-git-branching-model/#the-main-branches)
