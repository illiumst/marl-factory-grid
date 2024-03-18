Testing
=======
In EDYS, tests are seamlessly integrated through environment hooks, mirroring the organization of rules, as explained in the README.md file.

Running tests
-------------
To include specific tests in your run, simply append them to the "tests" section within the configuration file.
If the test requires a specific entity in the environment (i.e the clean up test requires a TSPDirtAgent that can observe
and clean dirt in its environment), make sure to include it in the config file.

Writing tests
------------
If you intend to create additional tests, refer to the tests.py file for examples.
Ensure that any new tests implement the corresponding test class and make use of its hooks.
There are no additional steps required, except for the inclusion of your custom tests in the config file.
