#define CTYPE c64
#define RTYPE f64
#define CTYPE_IS_COMPLEX
#define CONCAT2(a, b) a##_##b
#define CONCAT(a, b) CONCAT2(a, b)
#define PREFIX z
#define FN(name) CONCAT(PREFIX, name)

#include "dipolar_conv_impl.inc"
