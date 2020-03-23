To conduct our experiments we utilized the following MINOTAUR version
https://github.com/Mirandatz/minotaur/tree/4dcb44fa6b5ed90f5aa0a40366d46ef0551d15bd

This version contains a minor bug (fixed in https://github.com/Mirandatz/minotaur/tree/447551fb5d575e68c517fa5559289643cb4d6666)
which generates poorly written .csv headers.
Specifically, the output mechanism doesn't specify which metrics
are related to the "train dataset" and which are related to the "test dataset".
But it is possible to identify which is which.
The first set of metrics is from the "train dataset" and the second one from the test dataset.
To exemplify, consider the following .csv header:
Generation Number,Individual Id,Individual Parent Id,MultiLabelFScore,RuleCount,MultiLabelFScore,RuleCount,Rules
Columns with (0-based) index 3 and 4 are from the "train dataset",
while columns with index 5 and 6 are from the "test dataset".