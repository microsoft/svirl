# Python
[PEP Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

## Naming Conventions

1. Use a naming style that follows `lower_case_with_underscores`.

2. Function names should be lowercase and separated by underscores for clarity.

3. Class names should follow `CapWords` format (also known as `CamelCase`).

4. `mixedCase` is not allowed.

5. `_single_leading_underscore`: weak internal use of variable. For instance, when a module is imported like `from M import *`, the objects with leading underscores are not imported.

6. `single_trailing_underscore_`: use to avoid conflicts with Python keyword (for e.g., `class_` instead of `class`).

7. Type variable names should use `CapWords` format.

8. Constants are written in all capital letters with underscores separating the words: E.g., `NWARPS`, `TOTAL_SUM`.


## Indents

1. Use 4 spaces for indentation.

2. Two empty lines between functions/methods.


## Comments

1. Should begin with capital (lowercase?) letter.


# C/CUDA
[PEP Style Guide for C Code](https://www.python.org/dev/peps/pep-0007/)


## Naming Conventions

1. CUDA seems to follow `mixedWords` format, but we will use the same convention as Python: function names should be lowercase and separated by underscores for clarity.

2. Function definition: The CUDA keywords used in function declaration should be placed in a separate line like this:
```
__device__ __inline__
void foo(int *a)
{
    return;
}
```

3. Use `}` (not `};`) after function declaration.

4. Two empty lines between functions.

5. Use `{` in the same line as preceeding `for`- and `if`-statements. Use `{` preceeding function declaration at the next line, e.g.,
```
void foo(int *a)
{
    if (a == NULL) {
        return;
    }
}
```
The following syntax is allowed for functions with a long argument list,
```
void foo(
    int arg0,
    int *arg1,
    int *arg2
) {
    return;
}
```
Note, `{`-rule can be changed in future.
