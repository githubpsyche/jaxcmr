# dev directory

I don't like Github Issues for tracking my development process. I like to keep my notes in markdown files in a directory that I can easily access and search. It's also nice to write notebooks or scripts sometimes when thinking about a problem. This directory is for tracking the development process of the codebase.

## Organization

Rough classification for how I'm organizing this directory.

### Archive
Stuff from this directory that I don't want to delete but don't want to see -- is no longer "live" with respect to current codebase or priorities. When I "remove" something from another part of this directory, I either delete it or move it here.

### Bugs
This subdirectory identifies bugs in the codebase. Each filename should clearly describe what the bug is and where it is located in the codebase. File body can include a description of the bug and any other research about it. Remove the file if the bug is fixed.

Bugs are strictly defined as things that are not working as intended. If something is working as intended but could be improved, it should be described in the `features` subdirectory.

### Design
This subdirectory contains proposals for design-level changes to the codebase -- mostly refactorings or modifications of types/protocols instead of new functionality or bug fixes. Each filename should clearly describe what the proposed change is and where it is located in the codebase. The file body should include a description of the proposed change, why it is important, and how it could be implemented. Remove the file if the proposal is implemented or rejected -- possibly include details in project documentation or a design document.

### Features
This subdirectory contains descriptions of features that could be added to the codebase. Each feature should be described in a markdown file with a clear title and a clear description of the feature. The description should include a description of the feature, why it is important, and how it could be implemented.

A lot of overlap with `design` subdirectory. Usually, `features` is for more concrete, smaller-scale changes, while `design` is for larger-scale changes. Another way to think about it is that `design` proposals frequently involve a lot of refactoring and focus on making it easier to add new features in the future, while `features` proposals are more about adding new functionality under the current codebase's design.

### Issues
Usually a miscellaneous directory for things that don't fit or haven't been classified yet. I should periodically review and clean this directory to ensure that it is up-to-date and that all issues are accurate and easy to understand. It should usually be empty. I might rename to misc but that's potentially misleading if an item is just waiting to be sorted.

Same naming and body convention as `bugs` subdirectory.

### Questions
Just stuff I want to remember to ask about. By sticking them here, I hopefully give myself space to work on something else without worrying I won't remember later. I should periodically review and clean this directory to ensure that it is up-to-date and that all questions are accurate and easy to understand. It should usually be empty.

### Prototyping
There's the codebase, and then there's the stuff I use this codebase to do (i.e., research). This is where I keep notes on research that I'm doing, usually defining tasks in terms of generating a report or a notebook, and component models or other codebase features that I need to implement to do the research. Remove once research is complete or no longer relevant. Or refactor into a paper draft or something.

In practice, it can be any code that I want to keep around but set apart from actual codebase. Name should not so much describe what the code does, but what the code is for -- as specific as possible in the context. Frequently it's a demo of a specific feature or a test of a specific bug. Sometimes it's just a workspace for a feature in progress.

I have outer "notebooks" and "projects" directory that contains template notebooks for generating reports using codebase functionality. If I clean up a notebook for re-use in this way, or to share with someone else, I'll move it to one of those directories.

Multi-file research or code that needs preservation should be moved to the `projects` directory. This is just for small-scale/one-off research that I'm prototyping or testing.


### Performance
This subdirectory contains descriptions of performance improvements that could be made to the codebase. Each performance improvement should be described in a markdown file with a clear title and a clear description of the improvement. The description should include a description of the improvement, why it is important, and how it could be implemented. Body should also include any research about the performance improvement.

### Policies
This subdirectory contains descriptions of static development policies that I want to enforce in the codebase. Each policy should be described in a markdown file with a clear title and a clear description of the policy. The description should include a description of the policy, why it is important, why I chose it. 

Some overlap with `design` subdirectory. Design documents are to strategy as policies are to tactics. Policies are more reminders about how to implement designs, while design documents are more about what to implement. I usually remove stuff from this directory once I'm confident that the policy is now established tradition for development.

### TODO
Day-to-day notes on what I need to do. Clean up after big shifts in task frame. Otherwise, having a memory of what I've done in the last several work sessions is useful.

Archiving all this stuff here makes it easier to not lose track of it. 

### Anything Missing?
Might be nice to build a set of literature notes here. Also, it's weird to do so much research here if actual projects will get their own repository. Whatever.

