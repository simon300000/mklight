// Fixed-order resident collectives. Scalar word bindings permit element-aligned
// views without assuming float2 alignment. Parameter words are encoder-owned.
#include "mklight_common.h"

static inline float2 mkl_c_load(device const uint* a, uint i, uint words) {
    return float2(as_type<float>(a[i*words]), words==2u ? as_type<float>(a[i*words+1u]) : 0.0f);
}
static inline void mkl_c_store(device uint* a,uint i,float2 v) {
    a[2u*i]=as_type<uint>(v.x);a[2u*i+1u]=as_type<uint>(v.y);
}
static inline float2 mkl_c_uint(uint i) {
    return mkl_df_add(float2(float(i>>16)*65536.0f,0.0f),float2(float(i&65535u),0.0f));
}
static inline uint mkl_c_fkey(float x) {
    uint b=as_type<uint>(x);
    if ((b&0x7fffffffu)==0u) b=0u;
    if ((b&0x7fffffffu)>0x7f800000u) return 0xffffffffu;
    return (b&0x80000000u)!=0u ? ~b : (b^0x80000000u);
}
static inline int mkl_c_cmp(float2 a,float2 b) {
    bool an=isnan(a.x)||isnan(a.y),bn=isnan(b.x)||isnan(b.y);
    if(an||bn)return an==bn?0:(an?1:-1);
    uint ah=mkl_c_fkey(a.x),bh=mkl_c_fkey(b.x);
    if(ah!=bh) return ah<bh?-1:1;
    uint al=mkl_c_fkey(a.y),bl=mkl_c_fkey(b.y);
    return al==bl?0:(al<bl?-1:1);
}
static inline float2 mkl_c_ext(float2 a,float2 b,bool maximum) {
    if(isnan(a.x)) return a;
    if(isnan(b.x)) return b;
    int cmp=mkl_c_cmp(a,b);
    return (maximum ? cmp<0 : cmp>0)?b:a;
}
static inline float2 mkl_c_combine(float2 a,float2 b,uint mode,uint ext) {
    if(ext!=0u) return mkl_c_ext(a,b,ext==2u);
    return mode==MKL_REDUCE_PLAIN ? float2(a.x+b.x,0.0f) : mkl_df_add(a,b);
}
static inline float2 mkl_c_tree(threadgroup float2* a,uint tid,uint mode,uint ext) {
    for(uint step=128u;step>0u;step>>=1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<step) a[tid]=mkl_c_combine(a[tid],a[tid+step],mode,ext);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return a[0];
}
// Scan: n, first block, input kind (0 u32, 1 f32, 2 df).
kernel void mkl_c_scan(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
    device uint* sums [[buffer(2)]], constant uint* p [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],uint g [[threadgroup_position_in_grid]]) {
    threadgroup float2 v[256];threadgroup uint u[256];
    uint i=(p[1]+g)*256u+tid;
    bool integer=p[2]==0u;
    if(integer) u[tid]=i<p[0]?src[i]:0u;
    else v[tid]=i<p[0]?mkl_c_load(src,i,p[2]):float2(0.0f);
    for(uint step=1u;step<256u;step<<=1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint pu=0u;float2 pv(0.0f);
        if(tid>=step) {if(integer) pu=u[tid-step];else pv=v[tid-step];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid>=step) {if(integer) u[tid]=pu+u[tid];else v[tid]=mkl_df_add(pv,v[tid]);}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(i<p[0]) {if(integer) dst[i]=u[tid];else mkl_c_store(dst,i,v[tid]);}
    if(tid==255u) {if(integer) sums[p[1]+g]=u[tid];else mkl_c_store(sums,p[1]+g,v[tid]);}
}
// Prefix fix-up: n, first item, item count, integer flag.
kernel void mkl_c_scan_add(device uint* dst [[buffer(0)]],device const uint* sums [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint t [[thread_position_in_grid]]) {
    if(t>=p[2]) return;uint i=p[1]+t;if(i>=p[0]||i<256u) return;
    uint b=i/256u-1u;
    if(p[3]!=0u) dst[i]+=sums[b];
    else mkl_c_store(dst,i,mkl_df_add(mkl_c_load(sums,b,2u),mkl_c_load(dst,i,2u)));
}
// Generic initialisation / mask normalisation / shifted final scan copy.
// n, first item, item count, operation (0 mask,1 identity index,2 scan copy),
// output words, inclusive.
kernel void mkl_c_init(device const uint* src [[buffer(0)]],device uint* dst [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint t [[thread_position_in_grid]]) {
    if(t>=p[2]) return;uint i=p[1]+t;if(i>=p[0]) return;
    if(p[3]==0u) dst[i]=uint(src[i]!=0u);
    else if(p[3]==1u) dst[i]=i;
    else if(p[3]==3u) dst[i]=0u;
    else for(uint w=0u;w<p[4];++w) dst[i*p[4]+w]=(p[5]!=0u||i>0u)?src[(p[5]!=0u?i:i-1u)*p[4]+w]:0u;
}
// Compact: n,first,count,record words. The tail is cleared by a separate
// kernel, so scatter and tail writes cannot race.
kernel void mkl_c_compact_clear(device uint* dst [[buffer(0)]],device uint* ids [[buffer(1)]],
    device uint* count [[buffer(2)]],device const uint* prefix [[buffer(3)]],constant uint* p [[buffer(4)]],uint t [[thread_position_in_grid]]) {
    if(t>=p[2]) return;uint i=p[1]+t;if(i>=p[0]) return;
    uint kept=prefix[p[0]-1u];
    if(i>=kept) {ids[i]=0xffffffffu;for(uint w=0u;w<p[3];++w)dst[i*p[3]+w]=0u;}
    if(i==0u)count[0]=kept;
}
kernel void mkl_c_compact(device const uint* src [[buffer(0)]],device const uint* ids [[buffer(1)]],
    device const uint* mask [[buffer(2)]],device const uint* prefix [[buffer(3)]],device uint* dst [[buffer(4)]],
    device uint* outids [[buffer(5)]],constant uint* p [[buffer(6)]],uint t [[thread_position_in_grid]]) {
    if(t>=p[2]) return;uint i=p[1]+t;if(i>=p[0]||mask[i]==0u)return;
    uint j=prefix[i]-1u;outids[j]=ids[i];for(uint w=0u;w<p[3];++w)dst[j*p[3]+w]=src[i*p[3]+w];
}
static inline bool mkl_c_less(uint a,uint b,device const uint* keys,device const uint* ids,uint kind) {
    int cmp=0;
    if(kind==0u) {cmp=keys[a]==keys[b]?0:(keys[a]<keys[b]?-1:1);}
    else cmp=mkl_c_cmp(mkl_c_load(keys,a,kind),mkl_c_load(keys,b,kind));
    if(cmp!=0) return cmp<0;
    return ids[a]!=ids[b]?ids[a]<ids[b]:a<b;
}
// Merge ranks: n, first, count, width, key kind. Every original index has a
// unique total order, so each thread writes a unique destination.
kernel void mkl_c_merge(device const uint* keys [[buffer(0)]],device const uint* ids [[buffer(1)]],
    device const uint* src [[buffer(2)]],device uint* dst [[buffer(3)]],constant uint* p [[buffer(4)]],uint t [[thread_position_in_grid]]) {
    if(t>=p[2]) return;uint i=p[1]+t;if(i>=p[0])return;
    uint width=p[3],base=(i/width/2u)*width*2u,mid=base+min(width,p[0]-base),end=mid+min(width,p[0]-mid);
    bool left=i<mid;uint lo=left?mid:base,hi=left?end:mid,begin=lo;
    uint a=src[i];
    while(lo<hi) {uint j=lo+(hi-lo)/2u;if(mkl_c_less(src[j],a,keys,ids,p[4]))lo=j+1u;else hi=j;}
    uint rank=i-(left?base:mid)+lo-begin;dst[base+rank]=a;
}
// Gather: n,first,count,key words,record words.
kernel void mkl_c_gather(device const uint* keys [[buffer(0)]],device const uint* ids [[buffer(1)]],
    device const uint* vals [[buffer(2)]],device const uint* perm [[buffer(3)]],device uint* outkeys [[buffer(4)]],
    device uint* outids [[buffer(5)]],device uint* outvals [[buffer(6)]],constant uint* p [[buffer(7)]],uint t [[thread_position_in_grid]]) {
    if(t>=p[2])return;uint j=p[1]+t;if(j>=p[0])return;uint i=perm[j];outids[j]=ids[i];
    for(uint w=0u;w<p[3];++w)outkeys[j*p[3]+w]=keys[i*p[3]+w];
    for(uint w=0u;w<p[4];++w)outvals[j*p[4]+w]=vals[i*p[4]+w];
}
static inline uint mkl_c_groups(uint n) {return clamp(n/4096u+uint(n%4096u!=0u),1u,256u);}
// Segment parameters: n,channels,segments,first segment,segment count,
// max groups,terms,phase,kind,mode,value words,reference words,max length,
// first group,first term,reserved.
static inline uint mkl_c_extmode(constant uint* p,uint term) {
    if(p[8]==MKL_COLLECT_SUMMARY) return term%4u>=2u?term%4u-1u:0u;
    if(p[8]==MKL_COLLECT_MOMENTS2&&p[7]==1u&&term>=4u)return 2u;
    return 0u;
}
static inline uint mkl_c_mode(constant uint* p,uint term) {
    bool count=(p[8]==MKL_COLLECT_SUMMARY&&term%4u==1u)||(p[8]==MKL_COLLECT_MOMENTS2&&p[7]==0u&&term==0u);
    return count?MKL_REDUCE_DF:p[9];
}
static inline float2 mkl_c_identity(constant uint* p,uint ext) {
    return float2(ext==1u?INFINITY:(ext==2u&&p[8]!=MKL_COLLECT_MOMENTS2?-INFINITY:0.0f),0.0f);
}
static inline float2 mkl_c_term(device const uint* vals,device const uint* refs,device const uint* out,
    constant uint* p,uint seg,uint term,uint i,device const uint* mask,device const uint* weights,uint2 options) {
    if(options.x!=0u&&mask[i]!=p[15]) {uint ext=mkl_c_extmode(p,term);return mkl_c_identity(p,ext);}
    if(p[8]==MKL_COLLECT_SUM||p[8]==MKL_COLLECT_SOURCE_PRODUCTS)return mkl_c_load(vals,term*p[0]+i,p[10]);
    if(p[8]==MKL_COLLECT_SUMMARY)return term%4u==1u?float2(1.0f,0.0f):mkl_c_load(vals,(term/4u)*p[0]+i,p[10]);
    float2 x=mkl_c_load(vals,i,p[10]),y=mkl_c_load(vals,p[0]+i,p[10]),w=options.y!=0u?mkl_c_load(weights,i,options.y):(p[1]==3u?mkl_c_load(vals,2u*p[0]+i,p[10]):float2(1.0f,0.0f));
    if(p[11]!=0u){x=mkl_df_sub(x,mkl_c_load(refs,2u*seg,p[11]));y=mkl_df_sub(y,mkl_c_load(refs,2u*seg+1u,p[11]));}
    if(p[7]==0u) {if(term==0u)return float2(1.0f,0.0f);if(term==1u)return w;return mkl_df_mul(w,term==2u?x:y);}
    if(term==0u||term==4u){float2 rr=mkl_df_add(mkl_df_mul(x,x),mkl_df_mul(y,y));return term==0u?mkl_df_mul(w,rr):(w.x>0.0f?rr:float2(0.0f));}
    x=mkl_df_sub(x,mkl_c_load(out,seg*10u+2u,2u));y=mkl_df_sub(y,mkl_c_load(out,seg*10u+3u,2u));
    if(term==5u)return w.x>0.0f?mkl_df_add(mkl_df_mul(x,x),mkl_df_mul(y,y)):float2(0.0f);
    return mkl_df_mul(w,term==1u?mkl_df_mul(x,x):(term==2u?mkl_df_mul(x,y):mkl_df_mul(y,y)));
}
kernel void mkl_c_seg1(device const uint* vals [[buffer(0)]],device const uint* offsets [[buffer(1)]],
    device const uint* refs [[buffer(2)]],device const uint* out [[buffer(3)]],device uint* partials [[buffer(4)]],
    constant uint* p [[buffer(5)]],device const uint* mask [[buffer(6)]],device const uint* weights [[buffer(7)]],constant uint2& options [[buffer(8)]],uint3 lid [[thread_position_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    uint tid=lid.x;
    threadgroup float2 tree[256];
    uint seg=p[3]+group.z,term=p[14]+group.y,g=p[13]+group.x;
    uint a=offsets[seg],b=offsets[seg+1u];bool valid=a<=b&&b<=p[0]&&b-a<=p[12];
    uint n=valid?b-a:0u,G=mkl_c_groups(n),ext=mkl_c_extmode(p,term),mode=mkl_c_mode(p,term);
    float2 acc=mkl_c_identity(p,ext);
    float kh=0.0f,kc=0.0f;
    if(g<G)for(uint i=g*256u+tid;i<n;i+=G*256u) {
        float2 v=mkl_c_term(vals,refs,out,p,seg,term,a+i,mask,weights,options);
        if(ext!=0u)acc=mkl_c_ext(acc,v,ext==2u);
        else if(mode==MKL_REDUCE_KAHAN){mkl_neumaier_add(kh,kc,v.x);if(v.y!=0.0f)mkl_neumaier_add(kh,kc,v.y);}
        else if(mode==MKL_REDUCE_PLAIN)acc.x+=v.x+v.y;
        else acc=mkl_df_add(acc,v);
    }
    if(ext==0u&&mode==MKL_REDUCE_KAHAN)acc=mkl_two_sum(kh,kc);
    tree[tid]=acc;float2 value=mkl_c_tree(tree,tid,mode,ext);
    if(tid==0u)mkl_c_store(partials,((group.z*p[6]+term)*p[5]+g),value);
}
kernel void mkl_c_seg2(device const uint* partials [[buffer(0)]],device const uint* offsets [[buffer(1)]],
    device uint* out [[buffer(2)]],device uint* status [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint2 lid [[thread_position_in_threadgroup]],uint2 group [[threadgroup_position_in_grid]]) {
    uint tid=lid.x;
    threadgroup float2 tree[256];uint seg=p[3]+group.y,term=group.x;
    uint a=offsets[seg],b=offsets[seg+1u];bool valid=a<=b&&b<=p[0]&&b-a<=p[12];
    uint G=mkl_c_groups(valid?b-a:0u),ext=mkl_c_extmode(p,term),mode=mkl_c_mode(p,term);
    float2 acc=mkl_c_identity(p,ext);
    if(tid<G) acc=mkl_c_combine(acc,mkl_c_load(partials,(group.y*p[6]+term)*p[5]+tid,2u),mode,ext);
    tree[tid]=acc;float2 v=mkl_c_tree(tree,tid,mode,ext);
    if(tid==0u) {
        if(term==0u)status[seg]=valid?0u:1u;
        if(!valid)v=float2(NAN,0.0f);
        if(p[8]==MKL_COLLECT_SUM)mkl_c_store(out,term*p[2]+seg,v);
        else if(p[8]==MKL_COLLECT_SUMMARY) {
            uint base=((term/4u)*p[2]+seg)*4u;
            mkl_c_store(out,base+term%4u,v);
        } else if(p[8]==MKL_COLLECT_SOURCE_PRODUCTS) {
            uint width=p[1]+p[1]*(p[1]+1u)/2u;mkl_c_store(out,seg*width+term,v);
        } else {
            // Phase zero publishes weight and weighted coordinate sums; a
            // separate finaliser forms the means without intergroup races.
            uint dest=p[7]==0u?term:term+4u;
            mkl_c_store(out,seg*10u+dest,v);
        }
    }
}
// Segment finalisation: same parameter block; one thread per segment.
kernel void mkl_c_seg_finish(device uint* out [[buffer(0)]],constant uint* p [[buffer(1)]],uint t [[thread_position_in_grid]]) {
    if(t>=p[4])return;uint seg=p[3]+t;
    if(p[8]==MKL_COLLECT_MOMENTS2) {
        float2 w=mkl_c_load(out,seg*10u+1u,2u);
        for(uint j=2u;j<4u;++j) {float2 v=w.x>0.0f?mkl_df_div(mkl_c_load(out,seg*10u+j,2u),w):float2(NAN,0.0f);mkl_c_store(out,seg*10u+j,v);}
    } else {
        uint c=p[1],base=seg*(c+c*(c+1u)/2u),dest=base+c;
        for(uint a=0u;a<c;++a)for(uint b=a;b<c;++b)mkl_c_store(out,dest++,mkl_df_mul(mkl_c_load(out,base+a,2u),mkl_c_load(out,base+b,2u)));
    }
}
// QR parameters: rows,cols,batches,k,stage,groups,first group,first column,
// first batch,column count,first row,row count,input words,batch count,0,aux stride.
static inline uint mkl_c_qf(uint b,uint i,uint j,constant uint* p) {return (b*(p[0]+p[1]+2u)+i)*p[1]+j;}
static inline bool mkl_c_qa(device const uint* diag,uint b,constant uint* p) {
    return as_type<float>(diag[(b*4u+3u)*2u])==0.0f;
}
kernel void mkl_c_qr_init(device const uint* a [[buffer(0)]],device const uint* rhs [[buffer(1)]],
    device uint* work [[buffer(2)]],device uint* factors [[buffer(3)]],device uint* diag [[buffer(4)]],
    device uint* aux [[buffer(5)]],constant uint* p [[buffer(6)]],uint t [[thread_position_in_grid]]) {
    uint total=p[11];if(t>=total)return;uint idx=p[10]+t,m=p[0],n=p[1];
    uint row=idx%m,col=(idx/m)%(n+1u),b=idx/(m*(n+1u));
    float2 v=col==n?mkl_c_load(rhs,b*m+row,p[12]):mkl_c_load(a,(b*m+row)*n+col,p[12]);
    mkl_c_store(work,idx,v);
    if(row==0u&&col<n)mkl_c_store(factors,mkl_c_qf(b,m+n+1u,col,p),mkl_c_uint(col));
    if(row==0u&&col==n) {
        for(uint j=0u;j<4u;++j)mkl_c_store(diag,b*4u+j,float2(j==1u?INFINITY:0.0f,0.0f));
        for(uint j=0u;j<p[15];++j)mkl_c_store(aux,b*p[15]+j,float2(0.0f));
    }
}
kernel void mkl_c_qr_reduce1(device const uint* work [[buffer(0)]],device const uint* factors [[buffer(1)]],
    device const uint* aux [[buffer(2)]],device uint* partial [[buffer(3)]],constant uint* p [[buffer(4)]],
    uint3 lid [[thread_position_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    uint tid=lid.x;
    threadgroup float2 tree[256];uint m=p[0],n=p[1],b=p[8]+group.z,j=p[7]+group.y,g=p[6]+group.x;
    float2 scale=mkl_c_load(aux,b*p[15]+j,2u),v(0.0f);
    for(uint i=g*256u+tid;i<m;i+=p[5]*256u) {
        float2 a=mkl_c_load(work,(b*(n+1u)+j)*m+i,2u),term;
        if(p[4]==0u)term=float2(abs(a.x),0.0f);
        else if(p[4]==1u) {float2 s=scale.x==0.0f?float2(0.0f):mkl_df_div(a,scale);term=mkl_df_mul(s,s);}
        else term=mkl_df_mul(mkl_c_load(factors,mkl_c_qf(b,i,p[3],p),2u),a);
        v=p[4]==0u?mkl_c_ext(v,term,true):mkl_df_add(v,term);
    }
    tree[tid]=v;v=mkl_c_tree(tree,tid,MKL_REDUCE_DF,p[4]==0u?2u:0u);
    if(tid==0u)mkl_c_store(partial,((group.z*(n+1u)+j)*p[5]+g),v);
}
kernel void mkl_c_qr_reduce2(device const uint* partial [[buffer(0)]],device uint* aux [[buffer(1)]],
    constant uint* p [[buffer(2)]],uint2 lid [[thread_position_in_threadgroup]],uint2 group [[threadgroup_position_in_grid]]) {
    uint tid=lid.x;
    threadgroup float2 tree[256];uint n=p[1],b=p[8]+group.y,j=p[7]+group.x;
    float2 v(0.0f);if(tid<p[5])v=mkl_c_combine(v,mkl_c_load(partial,(group.y*(n+1u)+j)*p[5]+tid,2u),MKL_REDUCE_DF,p[4]==0u?2u:0u);
    tree[tid]=v;v=mkl_c_tree(tree,tid,MKL_REDUCE_DF,p[4]==0u?2u:0u);
    if(tid==0u) {
        if(p[4]==1u) v=mkl_df_mul(mkl_df_sqrt(v),mkl_c_load(aux,b*p[15]+j,2u));
        mkl_c_store(aux,b*p[15]+(p[4]==0u?0u:n+1u)+j,v);
    }
}
// Equilibrate all columns and retain original scales in the factor record.
kernel void mkl_c_qr_scale(device uint* work [[buffer(0)]],device uint* factors [[buffer(1)]],
    device const uint* aux [[buffer(2)]],constant uint* p [[buffer(3)]],uint t [[thread_position_in_grid]]) {
    if(t>=p[11])return;uint idx=p[10]+t,m=p[0],n=p[1],i=idx%m,j=(idx/m)%n,b=idx/(m*n);
    float2 scale=mkl_c_load(aux,b*p[15]+n+1u+j,2u),v=mkl_c_load(work,(b*(n+1u)+j)*m+i,2u);
    if(scale.x!=0.0f)v=mkl_df_div(v,scale);
    mkl_c_store(work,(b*(n+1u)+j)*m+i,v);
    if(i==0u)mkl_c_store(factors,mkl_c_qf(b,m+n,j,p),scale);
}
// Small fixed-order pivot/control step; no host readback of factors or status.
kernel void mkl_c_qr_pivot(device uint* factors [[buffer(0)]],device uint* diag [[buffer(1)]],
    device uint* aux [[buffer(2)]],constant uint* p [[buffer(3)]],uint t [[thread_position_in_grid]]) {
    if(t>=p[13])return;uint b=p[8]+t,m=p[0],n=p[1],k=p[3],off=b*p[15];
    if(!mkl_c_qa(diag,b,p))return;
    uint best=k;
    for(uint j=k+1u;j<n;++j) {
        float2 a=mkl_c_load(aux,off+n+1u+j,2u),v=mkl_c_load(aux,off+n+1u+best,2u);
        int cmp=mkl_c_cmp(a,v);
        if(cmp>0||(cmp==0&&mkl_c_load(factors,mkl_c_qf(b,m+n+1u,j,p),2u).x<mkl_c_load(factors,mkl_c_qf(b,m+n+1u,best,p),2u).x))best=j;
    }
    mkl_c_store(aux,off+3u*(n+1u),mkl_c_uint(best));
    float2 d=mkl_c_load(aux,off+n+1u+best,2u);
    if(k==0u)mkl_c_store(aux,off+3u*(n+1u)+1u,d);
    float tolerance=as_type<float>(p[14]);
    if(!isfinite(d.x)||!isfinite(d.y)) {mkl_c_store(diag,b*4u+3u,float2(2.0f,0.0f));return;}
    if(d.x<=mkl_df_mul_f32(mkl_c_load(aux,off+3u*(n+1u)+1u,2u),tolerance).x) {mkl_c_store(diag,b*4u+3u,float2(1.0f,0.0f));return;}
    mkl_c_store(diag,b*4u,mkl_c_uint(k+1u));
    mkl_c_store(aux,off+n+1u+k,d);
    for(uint i=0u;i<k;++i) {
        float2 a=mkl_c_load(factors,mkl_c_qf(b,m+i,k,p),2u),c=mkl_c_load(factors,mkl_c_qf(b,m+i,best,p),2u);
        mkl_c_store(factors,mkl_c_qf(b,m+i,k,p),c);mkl_c_store(factors,mkl_c_qf(b,m+i,best,p),a);
    }
    float2 a=mkl_c_load(factors,mkl_c_qf(b,m+n+1u,k,p),2u),c=mkl_c_load(factors,mkl_c_qf(b,m+n+1u,best,p),2u);
    mkl_c_store(factors,mkl_c_qf(b,m+n+1u,k,p),c);mkl_c_store(factors,mkl_c_qf(b,m+n+1u,best,p),a);
    mkl_c_store(factors,mkl_c_qf(b,m+k,k,p),d);
}
kernel void mkl_c_qr_swap(device uint* work [[buffer(0)]],device uint* factors [[buffer(1)]],
    device const uint* diag [[buffer(2)]],device const uint* aux [[buffer(3)]],constant uint* p [[buffer(4)]],uint t [[thread_position_in_grid]]) {
    if(t>=p[11])return;uint idx=p[10]+t,m=p[0],n=p[1],i=idx%m,b=idx/m,k=p[3];
    if(!mkl_c_qa(diag,b,p))return;uint best=uint(mkl_c_load(aux,b*p[15]+3u*(n+1u),2u).x);
    float2 a=mkl_c_load(work,(b*(n+1u)+k)*m+i,2u),c=mkl_c_load(work,(b*(n+1u)+best)*m+i,2u);
    mkl_c_store(work,(b*(n+1u)+k)*m+i,c);if(best!=k)mkl_c_store(work,(b*(n+1u)+best)*m+i,a);
    mkl_c_store(factors,mkl_c_qf(b,i,k,p),mkl_df_div(c,mkl_c_load(aux,b*p[15]+n+1u+k,2u)));
}
kernel void mkl_c_qr_update(device uint* work [[buffer(0)]],device uint* factors [[buffer(1)]],
    device const uint* diag [[buffer(2)]],device uint* aux [[buffer(3)]],constant uint* p [[buffer(4)]],uint t [[thread_position_in_grid]]) {
    if(t>=p[11])return;uint idx=p[10]+t,m=p[0],n=p[1],k=p[3],i=idx%m,j=(idx/m)%(n-k)+k+1u,b=idx/(m*(n-k));
    if(!mkl_c_qa(diag,b,p))return;
    float2 dot=mkl_c_load(aux,b*p[15]+n+1u+j,2u),q=mkl_c_load(factors,mkl_c_qf(b,i,k,p),2u),v=mkl_c_load(work,(b*(n+1u)+j)*m+i,2u);
    mkl_c_store(work,(b*(n+1u)+j)*m+i,mkl_df_sub(v,mkl_df_mul(q,dot)));
    if(i==0u) {
        if(j<n) {uint dest=mkl_c_qf(b,m+k,j,p);mkl_c_store(factors,dest,mkl_df_add(mkl_c_load(factors,dest,2u),dot));}
        else {uint dest=b*p[15]+2u*(n+1u)+k;mkl_c_store(aux,dest,mkl_df_add(mkl_c_load(aux,dest,2u),dot));}
    }
}
static inline float2 mkl_c_absdf(float2 x) {return x.x<0.0f?-x:x;}
kernel void mkl_c_qr_finish(device const uint* factors [[buffer(0)]],device uint* diag [[buffer(1)]],
    device uint* aux [[buffer(2)]],device uint* solution [[buffer(3)]],constant uint* p [[buffer(4)]],uint t [[thread_position_in_grid]]) {
    if(t>=p[13])return;uint b=p[8]+t,m=p[0],n=p[1],off=b*p[15];
    float2 residual=mkl_c_load(aux,off+2u*n+1u,2u);
    mkl_c_store(diag,b*4u+2u,residual);
    if(!isfinite(residual.x))mkl_c_store(diag,b*4u+3u,float2(2.0f,0.0f));
    if(!mkl_c_qa(diag,b,p)) {for(uint j=0u;j<n;++j)mkl_c_store(solution,b*n+j,float2(NAN,0.0f));return;}
    bool finite_solution=true;
    // aux first block is x; the second block becomes inverse row sums.
    for(uint ri=n;ri>0u;--ri) {
        uint i=ri-1u;float2 v=mkl_c_load(aux,off+2u*(n+1u)+i,2u);
        for(uint j=i+1u;j<n;++j)v=mkl_df_sub(v,mkl_df_mul(mkl_c_load(factors,mkl_c_qf(b,m+i,j,p),2u),mkl_c_load(aux,off+j,2u)));
        v=mkl_df_div(v,mkl_c_load(factors,mkl_c_qf(b,m+i,i,p),2u));mkl_c_store(aux,off+i,v);
        uint pivot=uint(mkl_c_load(factors,mkl_c_qf(b,m+n+1u,i,p),2u).x);
        float2 coefficient=mkl_df_div(v,mkl_c_load(factors,mkl_c_qf(b,m+n,pivot,p),2u));
        finite_solution=finite_solution&&isfinite(coefficient.x)&&isfinite(coefficient.y);
        mkl_c_store(solution,b*n+pivot,coefficient);
    }
    if(!finite_solution){mkl_c_store(diag,b*4u+3u,float2(2.0f,0.0f));for(uint j=0u;j<n;++j)mkl_c_store(solution,b*n+j,float2(NAN,0.0f));return;}
    float2 normr(0.0f),norminv(0.0f);
    for(uint i=0u;i<n;++i) {
        float2 row(0.0f);for(uint j=i;j<n;++j)row=mkl_df_add(row,mkl_c_absdf(mkl_c_load(factors,mkl_c_qf(b,m+i,j,p),2u)));
        normr=mkl_c_ext(normr,row,true);mkl_c_store(aux,off+n+1u+i,float2(0.0f));
    }
    for(uint col=0u;col<n;++col)for(uint ri=col+1u;ri>0u;--ri) {
        uint i=ri-1u;float2 v(i==col?1.0f:0.0f,0.0f);
        for(uint j=i+1u;j<=col;++j)v=mkl_df_sub(v,mkl_df_mul(mkl_c_load(factors,mkl_c_qf(b,m+i,j,p),2u),mkl_c_load(aux,off+j,2u)));
        v=mkl_df_div(v,mkl_c_load(factors,mkl_c_qf(b,m+i,i,p),2u));mkl_c_store(aux,off+i,v);
        uint dest=off+n+1u+i;mkl_c_store(aux,dest,mkl_df_add(mkl_c_load(aux,dest,2u),mkl_c_absdf(v)));
    }
    for(uint i=0u;i<n;++i)norminv=mkl_c_ext(norminv,mkl_c_load(aux,off+n+1u+i,2u),true);
    mkl_c_store(diag,b*4u+1u,mkl_df_mul(normr,norminv));
}
