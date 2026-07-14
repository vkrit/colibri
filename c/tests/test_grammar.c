#include <stdio.h>
#include <string.h>

#include "../grammar.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

/* feed the walker with a string; return how many bytes were consumed */
static int feed(GrState *S, const char *s){
    int n=0;
    while(s[n]){ if(gr_accept(S,(unsigned char)s[n])!=1) break; n++; }
    return n;
}

int main(void){
    static Grammar G;
    GrState S;
    char buf[512];

    /* literal: the whole prefix is forced, then the parse terminates */
    CHECK(gr_parse(&G,"root ::= \"{\\\"id\\\":\"")==0);
    gr_state_init(&S,&G);
    CHECK(S.alive);
    CHECK(gr_forced(&S,buf,sizeof buf)==6);
    CHECK(!memcmp(buf,"{\"id\":",6));
    CHECK(feed(&S,"{\"id\":")==6);
    unsigned char m[32]; int end;
    CHECK(gr_admissible(&S,m,&end)==0 && end==1);        /* parse complete: only EOS */
    gr_free(&G);

    /* alternate: forcing stops at the branch and resumes after it */
    CHECK(gr_parse(&G,"root ::= \"a\" (\"b\" | \"c\") \"d\"")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==1 && buf[0]=='a');
    CHECK(feed(&S,"ab")==2);
    CHECK(gr_forced(&S,buf,sizeof buf)==1 && buf[0]=='d');
    gr_free(&G);

    /* enum, #48 style: the quote is forced, then from the first byte the whole value */
    CHECK(gr_parse(&G,
        "root ::= \"\\\"\" val \"\\\"\"\n"
        "val  ::= \"no_fit\" | \"partial_fit\" | \"good_fit\"")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==1 && buf[0]=='\"');
    CHECK(feed(&S,"\"n")==2);
    CHECK(gr_forced(&S,buf,sizeof buf)==6 && !memcmp(buf,"o_fit\"",6));
    gr_free(&G);

    /* classes with ranges, star: no forcing where the grammar branches */
    CHECK(gr_parse(&G,"root ::= \"a\" [0-9]* \"b\"")==0);
    gr_state_init(&S,&G);
    CHECK(feed(&S,"a")==1);
    CHECK(gr_admissible(&S,m,&end)==11 && end==0);       /* 0-9 or b */
    CHECK(gr_forced(&S,buf,sizeof buf)==0);
    CHECK(feed(&S,"42b")==3);
    CHECK(gr_admissible(&S,m,&end)==0 && end==1);
    gr_free(&G);

    /* plus on a group: forcing stops where the parse can terminate */
    CHECK(gr_parse(&G,"root ::= (\"x\" \"\\n\")+")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==2 && buf[0]=='x' && buf[1]=='\n');
    CHECK(feed(&S,"x\n")==2);
    CHECK(gr_admissible(&S,m,&end)==1 && end==1);        /* can close or open a line */
    CHECK(gr_forced(&S,buf,sizeof buf)==0);
    gr_free(&G);

    /* postfix on a multi-byte literal: repeats the ENTIRE literal */
    CHECK(gr_parse(&G,"root ::= \"ab\"+ \"c\"")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==2 && !memcmp(buf,"ab",2));
    CHECK(feed(&S,"ab")==2);
    CHECK(gr_admissible(&S,m,&end)==2 && end==0);        /* 'a' (repeats) or 'c' (closes) */
    CHECK(feed(&S,"abc")==3);
    CHECK(gr_admissible(&S,m,&end)==0 && end==1);
    gr_free(&G);

    /* negated class: the union with the closure covers all bytes -> no forcing */
    CHECK(gr_parse(&G,"root ::= \"\\\"\" [^\"]* \"\\\"\"")==0);
    gr_state_init(&S,&G);
    CHECK(feed(&S,"\"")==1);
    CHECK(gr_admissible(&S,m,&end)==256 && end==0);
    CHECK(feed(&S,"ciao \\ mondo\"")==13);
    CHECK(gr_admissible(&S,m,&end)==0 && end==1);
    gr_free(&G);

    /* desync: a byte that is not admitted does NOT change the state */
    CHECK(gr_parse(&G,"root ::= \"ab\"")==0);
    gr_state_init(&S,&G);
    CHECK(gr_accept(&S,'x')==0);
    CHECK(gr_accept(&S,'a')==1 && gr_accept(&S,'b')==1);
    gr_free(&G);

    /* hexadecimal escape and comments; rules over multiple lines */
    CHECK(gr_parse(&G,
        "# test grammar\n"
        "root ::= \"\\x41\"   # an A\n"
        "         [\\x30-\\x32]\n")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==1 && buf[0]=='A');
    CHECK(feed(&S,"A1")==2);
    CHECK(gr_admissible(&S,m,&end)==0 && end==1);
    gr_free(&G);

    /* parse errors: undefined rule, missing root, ')' out of place */
    CHECK(gr_parse(&G,"root ::= foo")!=0);
    CHECK(gr_parse(&G,"a ::= \"x\"")!=0);
    CHECK(gr_parse(&G,"root ::= \"x\" )")!=0);

    /* left recursion: parse ok, walker shuts down without blocking (fail-safe) */
    CHECK(gr_parse(&G,"root ::= root \"a\" | \"b\"")==0);
    gr_state_init(&S,&G);
    CHECK(!S.alive);
    CHECK(gr_forced(&S,buf,sizeof buf)==0);
    gr_free(&G);

    /* realistic NDJSON grammar (the #48 workload): long forced spans */
    CHECK(gr_parse(&G,
        "root ::= riga+\n"
        "riga ::= \"{\\\"id\\\":\\\"\" chiave \"\\\",\\\"fit_category\\\":\\\"\" cat \"\\\"}\" \"\\n\"\n"
        "chiave ::= [a-z0-9-]+\n"
        "cat  ::= \"no_fit\" | \"partial_fit\" | \"good_fit\"\n")==0);
    gr_state_init(&S,&G);
    CHECK(gr_forced(&S,buf,sizeof buf)==7 && !memcmp(buf,"{\"id\":\"",7));
    CHECK(feed(&S,"{\"id\":\"ocds-123\"")==16);
    int nf=gr_forced(&S,buf,sizeof buf);
    buf[nf]=0;
    CHECK(nf==17 && !strcmp(buf,",\"fit_category\":\""));
    CHECK(feed(&S,",\"fit_category\":\"p")==18);
    nf=gr_forced(&S,buf,sizeof buf); buf[nf]=0;
    CHECK(nf==13 && !strcmp(buf,"artial_fit\"}\n"));
    CHECK(feed(&S,"artial_fit\"}\n")==13);
    /* at the end of a line the parse can terminate (riga+): no forcing — the model can
     * stop here. Forcing resumes as soon as the model opens the next line. */
    CHECK(gr_admissible(&S,m,&end)==1 && end==1);
    CHECK(gr_forced(&S,buf,sizeof buf)==0);
    CHECK(feed(&S,"{")==1);
    CHECK(gr_forced(&S,buf,sizeof buf)==6 && !memcmp(buf,"\"id\":\"",6));
    gr_free(&G);

    puts("test_grammar: ok");
    return 0;
}
