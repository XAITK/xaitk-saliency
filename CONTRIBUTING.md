# Contributing to XAITK-Saliency

## Making a Contribution

Here we describe at a high level how to contribute to XAITK-Saliency. See the
[XAITK-Saliency README](README.md) file for additional information.

1. Navigate to the official XAITK-Saliency repository maintained
   [on GitHub](https://github.com/XAITK/xaitk-saliency).

2. Fork XAITK-Saliency into your GitHub user namespace and clone that onto your
   system.

3. Create a topic branch, edit files and create commits:

   ```
   $ git checkout -b <branch-name>
   $ <edit things>
   $ git add <file1> <file2> ...
   $ git commit
   ```

   - Included in your commits should be an addition to the
     `docs/release_notes/pending_release/` directory. This addition should be a
     `rst` file with a bullted list. THe list should be a short, descriptive
     summary of the update, feature or fix that was added. This is generally
     required for merger approval.

4. Push topic branch with commits to your fork in GitHub:

   ```
   $ git push origin HEAD -u
   ```

5. Visit the Kitware XAITK-Saliency GitHub, browse to the "Pull requests" tab
   and click on the "New pull request" button in the upper-right. Click on the
   "Compare across forks" link, browse to your fork and browse to the topic
   branch for the pull request. Finally, click the "Create pull request" button
   to create the request.

XAITK-Saliency uses GitHub for code review and GitHub Actions for continuous
testing. New pull requests trigger Continuous Integration workflows (CI) when
the merge target is the `master` or `release`-variant branch. All checks/tests
must pass before a PR can be merged by an individual with the appropriate
permissions.

We use Sphinx for manual and automatic API [documentation](docs). For spelling
and word usage guidance, see the "Style Sheet for XAITK Saliency Docs"
(`docs/misc/style_sheet.rst`).

### Jupyter Notebooks

When adding or modifying a Jupyter Notebook in this repository, consider the
following:

- Notebooks should be executable on Google Colab.
- Notebooks should be included in the appropriate CI workflow and be runnable in
  that environment in a timely manner.

#### Colab

XAITK-Saliency example notebooks have established a pattern of providing a
reference to, and support execution in, Google Colab. Notebooks should include,
near the bottom of their introduction section, a reference to itself on Google
Colab. See existing example notebooks for examples on what this looks like, e.g.
at the bottom of cell #1 in the `examples/OcclusionSaliency.ipynb` notebook.

Notebooks should also be runnable when executed in Google Colab. This often
requires a cell that performs `pip install ...` commands to bring in the
appropriate dependencies (including `xaitk-saliency` itself) into the at-first
empty Colab environment. This will also help with execution in the CI
environment as detailed next.

#### Notebook CI

This repository has set up a CI workflow to execute notebooks to ensure their
continued functionality, avoid bit-rot and avoid violation of established
use-cases. When contributing a Jupyter notebook, as an example or otherwise, a
reference should be added to this CI workflow
([located here ~L27](.github/workflows/ci-example-notebooks.yml)) to enable its
inclusion in the CI testing.

To that end, in developing the notebook, consider its execution in this CI
environment:

- should be concise in its execution time to not stall or time-out the CI
  workflow.
- should be light on computational resources to not exceed what is provided in
  the CI environment.

### Contribution Release Note Exceptions

When a new contribution is fixing a bug or minor issue with something that has
been recently contributed, it may be the case that no additional release notes
are needed since they would add redundancy to the document.

For example, let's say that a recent contribution added a feature `Foo` and an
appropriate release note for that feature. If a bug with that feature is quickly
noticed and fixed in a follow-on contribution that does not impact how the
feature is summarized in the release notes, then the release-notes check on that
follow-on contribution may be ignored by the reviewers of the contribution.

Generally, a reviewer will assume that a release note is required unless the
contributor makes a case that the check should be ignored. This will be
considered by reviewers on a case-by-case basis.

## Class Naming Philosophy

For classes that define a behavior, or perform a transformation of a subject, we
choose to follow the "Verb-Noun" style of class naming. The motivation to
prioritize the verb first is because of our heavy use of interface classes to
define API standards and how their primary source of implementation variability
is in "how" the verb is achieved. The noun subject of the verb usually describes
the input provided, or output returned, at runtime. Another intent of this take
on naming positively impacts intuitive understanding of class meaning and
function for user-facing interfaces and utilities.

Class names should represent the above as accurately, yet concisely as possible
and appropriately differentiate from other classes defined here. If there are
multiple classes that perform similar behaviors but on different subjects, it is
important to distinguish the subject of the verb in the naming. For example,
`PerturbImage` vs. `PerturbVideo` share the same verb, "perturb", but the
subject that is to be perturbed is noted as the differentiator.

Some concrete examples as can be found in this repository are:

- [`PerturbImage`](src/xaitk_saliency/interfaces/perturb_image.py)
  - verb: `Perturb`
  - noun: `Image` (input and output)
- [`GenerateClassifierConfidenceSaliency`](src/xaitk_saliency/interfaces/gen_classifier_conf_sal.py)
  - verb: `Generate`
  - noun(s): `ClassifierConfidence` (input) and `Saliency` (output)

A special case for class naming can be made for **concrete implementations**
derived from interface classes. This consideration aims to reduce repetitiveness
in implementation class names, since they are performing the same verb on the
same nouns but with different "hows." Even more concise concrete implementation
names are also beneficial in the interplay with plugin discovery and
configuration where class names *are* the keys for selection and are only
discoverable/configurable under the context of their parent interface classes.
In these cases it may be considered that the parent interface class(es) satisfy
the verb-noun naming rule, and the naming of the implementation class should
more reflect nature of the "how" specialization.

For an example of this exception case, let us consider the
`GenerateClassifierConfidenceSaliency` interface. With the interface being
moderately lengthy in name it can easily be seen how repetition of this with
additional verbiage to indicate specialization results is overly long names. The
implementation names used here for some implementations are `OcclusionScoring`
and `RISEScoring`. These drop the Verb-Noun aspects of the parent interface and
replace it with implementation specific nomenclature of "occlusion"
(method-descriptive) and "RISE" (acronym from source publication).
