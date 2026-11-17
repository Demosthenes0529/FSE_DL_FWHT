#define _GNU_SOURCE
#include "simon_transition.h"
#include <inttypes.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { uint64_t s[4]; } rng_state;

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64-k));
}

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static void rng_seed(rng_state *r, uint64_t seed) {
    for (int i=0;i<4;++i) r->s[i]=splitmix64(&seed);
}

static inline uint64_t rng_next(rng_state *r) {
    uint64_t result=rotl(r->s[1]*5,7)*9;
    uint64_t t=r->s[1]<<17;
    r->s[2]^=r->s[0]; r->s[3]^=r->s[1]; r->s[1]^=r->s[2]; r->s[0]^=r->s[3];
    r->s[2]^=t; r->s[3]=rotl(r->s[3],45);
    return result;
}

static int sampler_self_test(void) {
    unsigned worst=0;
    for(unsigned delta=0;delta<256;++delta){
        unsigned a[256]={0},b[256]={0};
        for(unsigned x=0;x<256;++x){
            uint8_t rl1=(uint8_t)((x<<1)|(x>>7));
            uint8_t rl3=(uint8_t)((x<<3)|(x>>5));
            uint8_t rl2=(uint8_t)((x<<2)|(x>>6));
            uint8_t fx=(uint8_t)(rl2^(rl1&rl3));
            unsigned y=x^delta;
            uint8_t yl1=(uint8_t)((y<<1)|(y>>7));
            uint8_t yl3=(uint8_t)((y<<3)|(y>>5));
            uint8_t yl2=(uint8_t)((y<<2)|(y>>6));
            uint8_t fy=(uint8_t)(yl2^(yl1&yl3));
            ++a[fx^fy];
            uint8_t dl1=(uint8_t)((delta<<1)|(delta>>7));
            uint8_t dl3=(uint8_t)((delta<<3)|(delta>>5));
            uint8_t dl2=(uint8_t)((delta<<2)|(delta>>6));
            uint8_t d=(uint8_t)(dl2^(rl1&dl3)^(dl1&rl3)^(dl1&dl3));
            ++b[d];
        }
        for(unsigned d=0;d<256;++d){unsigned e=a[d]>b[d]?a[d]-b[d]:b[d]-a[d];if(e>worst)worst=e;}
    }
    printf("SAMPLER_TEST exhaustive_8bit_max_count_error=%u status=%s\n",worst,worst?"FAIL":"PASS");
    return worst?1:0;
}

static void usage(const char *p){
    fprintf(stderr,"usage: %s --rounds R --samples-power P --seed HEX --output FILE [--family simon|simeck] [--word-size N] [--threads N] [--target-left HEX] [--target-right HEX]\n",p);
}

int main(int argc,char **argv){
    if(argc==2&&!strcmp(argv[1],"--self-test"))return sampler_self_test();
    int rounds=9,power=22,threads=1,word_size=64; unsigned and_a=1,and_b=8,xor_c=2; const char *family="simon"; uint64_t seed=UINT64_C(0x12802901); const char *outpath=NULL;
    uint64_t target_left=0,target_right=UINT64_C(0x0000000020000000);
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--rounds")&&i+1<argc)rounds=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--family")&&i+1<argc){family=argv[++i];if(!strcmp(family,"simeck")){and_a=0;and_b=5;xor_c=1;}else if(strcmp(family,"simon")){usage(argv[0]);return 2;}}
        else if(!strcmp(argv[i],"--word-size")&&i+1<argc)word_size=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--samples-power")&&i+1<argc)power=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--seed")&&i+1<argc)seed=strtoull(argv[++i],NULL,0);
        else if(!strcmp(argv[i],"--output")&&i+1<argc)outpath=argv[++i];
        else if(!strcmp(argv[i],"--threads")&&i+1<argc)threads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--target-left")&&i+1<argc)target_left=strtoull(argv[++i],NULL,0);
        else if(!strcmp(argv[i],"--target-right")&&i+1<argc)target_right=strtoull(argv[++i],NULL,0);
        else {usage(argv[0]);return 2;}
    }
    if(!outpath||rounds<1||rounds>64||power<1||power>62||threads<1||word_size<9||word_size>64){usage(argv[0]);return 2;}
    uint64_t word_mask=simon_word_mask((unsigned)word_size);
    target_left&=word_mask;target_right&=word_mask;
    uint64_t samples=UINT64_C(1)<<power;
    uint64_t (*counts)[64]=calloc((size_t)rounds+2,sizeof(*counts));
    uint64_t (*thread_counts)[64]=calloc((size_t)threads*(rounds+2),sizeof(*thread_counts));
    if(!counts||!thread_counts){perror("calloc");return 3;}
    omp_set_num_threads(threads);
    struct timespec ts0,ts1;clock_gettime(CLOCK_MONOTONIC,&ts0);
#pragma omp parallel
    {
        int tid=omp_get_thread_num(); rng_state rng; rng_seed(&rng,seed^(UINT64_C(0x9e3779b97f4a7c15)*(tid+1)));
        uint64_t (*local)[64]=thread_counts+(size_t)tid*(rounds+2);
#pragma omp for schedule(static)
        for(uint64_t n=0;n<samples;++n){
            uint64_t L=target_left,R=target_right;
            uint64_t v=R;while(v){unsigned b=__builtin_ctzll(v);++local[rounds][b];v&=v-1;}
            v=L;while(v){unsigned b=__builtin_ctzll(v);++local[rounds+1][b];v&=v-1;}
            for(int depth=1;depth<=rounds;++depth){
                uint64_t prevL=R;
                uint64_t prevR=(L^simon_sample_derivative_params_n(R,rng_next(&rng)&word_mask,(unsigned)word_size,and_a,and_b,xor_c))&word_mask;
                L=prevL;R=prevR;
                int original=rounds-depth;
                v=R;while(v){unsigned b=__builtin_ctzll(v);++local[original][b];v&=v-1;}
            }
        }
    }
    for(int t=0;t<rounds+2;++t)for(int b=0;b<word_size;++b)
        for(int th=0;th<threads;++th)counts[t][b]+=thread_counts[(size_t)th*(rounds+2)+t][b];
    clock_gettime(CLOCK_MONOTONIC,&ts1);
    FILE *f=fopen(outpath,"w");if(!f){perror(outpath);return 4;}
    fprintf(f,"window_index,reverse_depth,bit_position,active_count,activation_probability\n");
    for(int t=0;t<rounds+2;++t){
        int depth=rounds-t;
        for(int b=0;b<64;++b)fprintf(f,"%d,%d,%d,%" PRIu64 ",%.17g\n",t,depth,b,counts[t][b],(double)counts[t][b]/samples);
    }
    fclose(f);
    double sec=(ts1.tv_sec-ts0.tv_sec)+1e-9*(ts1.tv_nsec-ts0.tv_nsec);
    printf("SAMPLER_RESULT family=%s word_size=%d rounds=%d samples=2^%d seed=0x%016" PRIx64 " target=(0x%016" PRIx64 ",0x%016" PRIx64 ") threads=%d runtime=%.6f output=%s\n",family,word_size,rounds,power,seed,target_left,target_right,threads,sec,outpath);
    free(counts);free(thread_counts);return 0;
}
