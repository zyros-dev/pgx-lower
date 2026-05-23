#include <signal.h>
#include <stddef.h>

extern "C" {

const char* progname = "test_runtime";

typedef void (*pqsigfunc)(int);

pqsigfunc pqsignal(int signo, pqsigfunc func) {
    struct sigaction act{}, oact{};
    act.sa_handler = func;
    sigaction(signo, &act, &oact);
    return oact.sa_handler;
}

}
