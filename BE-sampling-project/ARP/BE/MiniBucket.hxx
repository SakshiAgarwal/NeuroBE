#ifndef MiniBUCKET_HXX_INCLUDED
#define MiniBUCKET_HXX_INCLUDED

#include <vector>

#include "Function.hxx"
#include "Utils/MiscUtils.hxx"


namespace BucketElimination
{

class MBEworkspace ;
class Bucket ;
class MiniBucket ;

typedef ARE::FnConstructor (*mboutputfncnstrctr)(MiniBucket & MB) ;
extern mboutputfncnstrctr MBoutputfncnstrctr ;

class MiniBucket
{
protected :
	MBEworkspace *_Workspace ;
	Bucket *_Bucket ;
	int32_t _IDX ; // index of this minibucket wrt the bucket
	int32_t _V ; // in case the bucket is created for a specic variable in bucket elimination, this is the var. -1 otherwise (e.g. in case of superbuckets).


public :
	inline MBEworkspace *Workspace(void) const { return _Workspace ; }
//	inline BucketElimination::Bucket *Bucket(void) const { return _Bucket ; }
	inline int32_t IDX(void) const { return _IDX ; }
	inline int32_t V(void) const { return _V ; }
	inline void SetIDX(int32_t IDX) { _IDX = IDX ; }
	inline void SetWorkspace(MBEworkspace *ws) { _Workspace = ws ; }
	inline void SetBucket(BucketElimination::Bucket *b) { _Bucket = b ; }

	// signature of this bucket is the set of all functions arguments. width is the size of signature.
protected :
	int32_t _Width ; // cardinality of signature of this bucket; this includes variables eliminated in this bucket.
	int32_t *_Signature ; // a union of the scopes of all functions (original/childbucketfunction(s)) in this bucket, including variables eliminated in this bucket.
	int32_t *_SortedSignature ; // signature sorted in non-decreasing order; computed to make AllowsFunction() check run faster.
	int32_t _SignatureArraySize ; // signature sorted in non-decreasing order; computed to make AllowsFunction() check run faster.
public :
	inline int32_t Width(void) const { return _Width ; }
	inline const int32_t *Signature(void) const { return _Signature ; }
	inline const int32_t *SortedSignature(void) const { return _SortedSignature ; }
	int32_t ComputeSignature(void) ;

	// variable(s) of the bucket that are eliminated when a bucket output fn is computed from functions of this bucket
protected :
	int32_t _nVars ;
	int32_t *_Vars ;
	int32_t _VarsSpace ;
public :
	inline int32_t nVars(void) const { return _nVars ; }
	inline int32_t Var(int32_t IDX) const { return _Vars[IDX] ; }
	inline int32_t *VarsArray(void) { return _Vars ; }
	inline int32_t AddVar(int32_t Var)
	{
		int32_t i ;
		for (i = 0 ; i < _nVars ; i++) {
			if (_Vars[i] == Var) 
				return 0 ;
			}
		if (_nVars >= _VarsSpace) {
			int32_t *v = new int32_t[_VarsSpace + 2] ;
			if (NULL == v) 
				return 1 ;
			for (i = 0 ; i < _nVars ; i++) 
				v[i] = _Vars[i] ;
			_VarsSpace += 2 ;
			if (NULL != _Vars) 
				delete [] _Vars ;
			_Vars = v ;
			}
		_Vars[_nVars++] = Var ;
		return 0 ;
	}

protected :
	int32_t _nFunctions ;
	ARE::Function **_Functions ;
	int32_t _FunctionsArraySize ;
public :
	inline int32_t nFunctions(void) const { return _nFunctions ; }
	inline ARE::Function *Function(int32_t IDX) const { return _Functions[IDX] ; }
	inline ARE::Function **FunctionsArray(void) { return _Functions ; }
	int32_t AddFunction(ARE::Function & F, std::vector<int32_t> & HelperArray) ;
	int32_t RemoveFunction(ARE::Function & F) ;

	// compute variables in F that are not in this MB; return number of vars.
	int32_t ComputeMissingVariables(ARE::Function & F, std::vector<int32_t> & MissingVars) ;

	// check if the given fn fits in this MB without breaking the limits.
	// 1=yes, 0=no, -1=function fails.
	int32_t AllowsFunction(ARE::Function & , int32_t MBmaxsize, std::vector<int32_t> & HelperArray) ;

	// ****************************************************************************************
	// Function generated by this bucket; its scope is _Signature-_V.
	// This function is generated by this bucket and belongs to this bucket.
	// However, after it is generated, it is placed in the parent-bucket.
	// Note that bucket function (as any other funcion), may have its table broken into pieces (blocks).
	// If the table is small, it may be kept in memory in its entirety (as 1 block); if it is large it may be broken into pieces (blocks) 
	// and each block stored as a table on the disk.
	// ****************************************************************************************

protected :

	ARE::Function *_OutputFunction ;

	// new function size (in number of elements) required to compute this bucket; this is the sum of child-function size + output fn size of this bucket.
	int64_t _ComputationNewFunctionSize ;

	// weighted mini-bucket weight of this minibucket
	double _WMBE_weight ;

public :

	inline ARE::Function & OutputFunction(void) { return *_OutputFunction ; }
	inline int64_t ComputationNewFunctionSize(void) const { return _ComputationNewFunctionSize ; }
	inline double & WMBE_weight(void) { return _WMBE_weight ; }

	int32_t CreateOutputFunction(ARE::FnConstructor Cnstrctr) ;

	// this creates the output function of this bucket and its scope; it does not fill in the table.
	// it sets the outputfn::Bucket member variable, but does not add it to the Bucket.
	int32_t ComputeOutputFunctionWithScopeWithoutTable(void) ;

	// processing complexity is the size of output table
	int64_t ComputeProcessingComplexity(void) ;

public :

	// output function has been computed.
	// to stuff, e.g. cleanup (release FTBs of all child buckets).
	int32_t NoteOutputFunctionComputationCompletion(void) ;

	// compute output function from scratch completely.
	// this fn is usually used when regular bucket elimination is used.
	// we build a MB table, by combining all MB functions. 
	// extras are : 
	//		-) adding avgMaxMarginal and subtract MaxMarginal.
	//		-) do power-sum with WMBEweight
	// the var(s) being eliminated are in the scope of avgMaxMarginal/MaxMarginal.
	virtual int32_t ComputeOutputFunction(int32_t varElimOperator, 
		ARE::Function *FU, ARE::Function *fU, // used when doing MomentMatching (product/max) (FU=fMaxMarginal fU=fAvgMaxMarginal) or CostShifting (product/sum)
		double WMBEweight // used when running WMBE
		) ;
	// Neural Network based version of ComputeOutputFunction()...
	// i.e. the output function is of type FunctionNN..
	virtual int32_t ComputeOutputFunction_NN(int32_t varElimOperator, 
		ARE::Function *FU, ARE::Function *fU, // used when doing MomentMatching (product/max) (FU=fMaxMarginal fU=fAvgMaxMarginal) or CostShifting (product/sum)
		double WMBEweight // used when running WMBE
		) ;

	// basic/general worker fn for eliminating a set of vars; we assume all ElimVars are in the minibucket signature.
	virtual int32_t ComputeOutputFunction(int32_t varElimOperator, 
		ARE::Function & OutputFN, 
		const int32_t *ElimVars, int32_t nElimVars, int32_t *TempSpaceForVars, 
		double WMBEweight // used when running WMBE
		) ;
	// eliminate all vars; output is const fn.
	virtual int32_t ComputeOutputFunction_EliminateAllVars(int32_t varElimOperator) ;

	inline void ApplyVarEliminationOperator(int32_t varElimOperator, bool ValueAreLogSpace, ARE_Function_TableType & V, ARE_Function_TableType v)
	{
		if (VAR_ELIMINATION_TYPE_SUM == varElimOperator) {
			if (ValueAreLogSpace) 
				LOG_OF_SUM_OF_TWO_NUMBERS_GIVEN_AS_LOGS(V, V, v)
			else 
				V += v ;
			}
		else if (VAR_ELIMINATION_TYPE_MAX == varElimOperator) 
			{ if (v > V) V = v ; }
		else if (VAR_ELIMINATION_TYPE_MIN == varElimOperator) 
			{ if (v < V) V = v ; }
	}

public :

	void Initalize(BucketElimination::Bucket & B, int32_t IDX) ;

	void Destroy(void) ;
	MiniBucket(void) ;
	MiniBucket(MBEworkspace & WS, int32_t IDX, int32_t V, ARE::FnConstructor Cnstrctr = ARE::FunctionConstructor) ;
	virtual ~MiniBucket(void) ;
//    int32_t nSamples = 1000;


} ;

} // namespace BucketElimination

#endif // MiniBUCKET_HXX_INCLUDED
