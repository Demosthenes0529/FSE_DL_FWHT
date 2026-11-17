#pragma once
#include <array>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace knot256_native {
using State=std::array<uint64_t,4>;constexpr uint64_t RC[63]={
0x01,0x02,0x04,0x08,0x10,0x21,0x03,0x06,0x0c,0x18,0x31,0x22,0x05,0x0a,0x14,0x29,
0x13,0x27,0x0f,0x1e,0x3d,0x3a,0x34,0x28,0x11,0x23,0x07,0x0e,0x1c,0x39,0x32,0x24,
0x09,0x12,0x25,0x0b,0x16,0x2d,0x1b,0x37,0x2e,0x1d,0x3b,0x36,0x2c,0x19,0x33,0x26,
0x0d,0x1a,0x35,0x2a,0x15,0x2b,0x17,0x2f,0x1f,0x3f,0x3e,0x3c,0x38,0x30,0x20};
inline uint64_t rol(uint64_t x,int n){n&=63;return n?(x<<n)|(x>>(64-n)):x;}
inline State xr(State a,const State&b){for(int i=0;i<4;i++)a[i]^=b[i];return a;}
inline State sbox(State x){uint64_t a=x[0],b=x[1],c=x[2],d=x[3],t1=~a,t2=b&t1,t3=c^t2,h=d^t3,t5=b|c,t6=d^t1,g=t5^t6,t8=b^d,t9=t3&t6,e=t8^t9,t11=g&t8,f=t3^t11;return {e,f,g,h};}
inline State linear(State x){x[1]=rol(x[1],1);x[2]=rol(x[2],8);x[3]=rol(x[3],25);return x;}
inline State inverse_linear(State x){x[1]=rol(x[1],63);x[2]=rol(x[2],56);x[3]=rol(x[3],39);return x;}
inline State encrypt_segment(State state,int start,int rounds){for(int r=start;r<start+rounds;r++){state[0]^=RC[r];state=linear(sbox(state));}return state;}
inline State parse(const std::string&text){std::string s=text;if(s.rfind("0x",0)==0)s=s.substr(2);if(s.size()>64)throw std::runtime_error("KNOT state wider than 256 bits");s=std::string(64-s.size(),'0')+s;State out{};for(int i=0;i<4;i++)out[i]=std::stoull(s.substr(16*i,16),nullptr,16);return out;}
inline std::string hex(const State&state){std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto x:state)out<<std::setw(16)<<x;return out.str();}
inline bool bit_at(const State&state,size_t bit){return (state[bit/64]>>(63-bit%64))&1;}
inline State panel_mask(uint32_t coefficient,const std::vector<int>&bits){State out{};for(size_t j=0;j<bits.size();j++)if(coefficient>>j&1)out[bits[j]/64]|=1ULL<<(63-bits[j]%64);return out;}
inline int parity(const State&a,const State&b){unsigned value=0;for(int i=0;i<4;i++)value^=__builtin_parityll(a[i]&b[i]);return value&1;}
}

