from setuptools import Extension, setup


module = Extension(
    "symnmfmodule",
    sources=["symnmfmodule.c", "symnmf.c", "matrix.c"],
    define_macros=[("SYMNMF_EXTENSION", "1")],
    libraries=["m"],
    extra_compile_args=["-Wall", "-Wextra", "-Werror"],
)


setup(
    name="symnmfmodule",
    version="1.0",
    description="Symmetric nonnegative matrix factorization",
    ext_modules=[module],
)
