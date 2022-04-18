#include <stdlib.h>
#include <memory.h>

#include <Function.hxx>
#include <Function-NN.hxx>
#include <Bucket.hxx>
#include <MBEworkspace.hxx>
#include <Bucket.hxx>
#include <MiniBucket.hxx>
#include <Sort.hxx>
#include "Utils/MersenneTwister.h"
#include <exception>
#include "Config.h"

//#include <thread>
#include <chrono>
//#include <iostream> //delete
//#include <fstream> //delete
static MTRand RNG ;

torch::optim::AdamOptions::AdamOptions(double lr, double eps) : lr_(lr), eps_(eps) {}

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int32_t BucketElimination::MiniBucket::ComputeOutputFunction_NN(int32_t varElimOperator, ARE::Function *FU, ARE::Function *fU, double WMBEweight)
{
    auto start = std::chrono::high_resolution_clock::now();
     int32_t i, k;

     bool convert_exp = true;
    if(0 == global_config.network.compare("net")){
        convert_exp=false;
    }
    else
        std::cout<<"masked_net.....";

    ARE::FunctionNN *fNN = dynamic_cast<ARE::FunctionNN *>(_OutputFunction) ;
    if (NULL == fNN)
        return 1 ;

    MBEworkspace *bews = dynamic_cast<MBEworkspace*>(_Workspace) ;
    if (NULL == bews)
        return ERRORCODE_generic ;
    ARE::ARP *problem = bews->Problem() ;
    if (NULL == problem)
        return ERRORCODE_generic ;
    if (_Width < 0) {
        if (0 != ComputeSignature())
            return ERRORCODE_generic ;
        if (_Width < 0)
            return ERRORCODE_generic ;
    }
    if (nVars() < 1)
        // nothing to do; should not happen; however, allow this to pass as ok; calling fn should be able to handle this.
        return 0 ;

    const int32_t w = Width() ;
    if (w < 0 || _nFunctions < 1)
        return 0 ;
    const int32_t *signature = Signature() ;

    // generate some number of random samples...
    Config config;
    int32_t nSamples;
    int n_val_samples, n_train_samples, n_test_samples,  total_samples;
    //double s = 0.2;
    double train_split = 0.8;

    //Approximate pseudo dimension
    int32_t w_in = fNN->N();
    int l = global_config.n_layers +1;
    float temp = ((l-1)*pow(w_in,2) + l*w_in + 4);
    float pd = temp*log(temp/l);
    float delta=0.001;

    // Estimate total number of samples
    nSamples = int((pd + log(1/delta))/global_config.epsilon);
    if (nSamples>1000000)
        nSamples=1000000;

    n_val_samples = int((1-train_split)*nSamples);
    n_train_samples = int((train_split)*nSamples);
    n_test_samples = 50000;

    total_samples= n_train_samples + n_val_samples + n_test_samples;
    std::cout<<"n train Samples----" <<n_train_samples<<"\n";

    fNN->train_samples = n_train_samples;
    fNN->val_samples = n_val_samples;
    fNN->test_samples = n_test_samples;

    std::vector<std::vector<float>> train_samples, val_samples,test_samples;
    train_samples.resize(n_train_samples, vector<float>(fNN->N()));
    val_samples.resize(n_val_samples, vector<float>(fNN->N()));
    test_samples.resize(n_test_samples, vector<float>(fNN->N()));

    std::vector<float> train_sample_values, val_sample_values, test_sample_values;
    train_sample_values.resize(n_train_samples);
    val_sample_values.resize(n_val_samples);
    test_sample_values.resize(n_test_samples);

    std::vector<int32_t> values_, vars_ ;
    values_.resize(problem->N(), 0) ;
    vars_.resize(problem->N(), 0) ;

    if (values_.size() != problem->N() || vars_.size() != problem->N())
        return 1 ;

    int32_t *vals = values_.data();
    int32_t *vars = vars_.data();

    k = problem->K(_V);
    for ( i = 0 ; i < fNN->N() ; ++i)
        vars[i] = fNN->Argument(i) ;
    vars[i] = _V ;

    int32_t ** samples_signiture;
    samples_signiture = new int32_t *[nSamples];
    for (int nn=0; nn<nSamples; nn++)
        samples_signiture[nn] = new int32_t[fNN->N()];
    float* sample_values = new float[nSamples];

    ARE_Function_TableType const_factor = bews->FnCombinationNeutralValue() ;
    int32_t nFNs = 0, n_fNN_in = 0 ;
    std::vector<ARE::Function *> flist ; flist.reserve(_nFunctions) ; if (flist.capacity() != _nFunctions) return 1 ;
    for (int32_t j = 0 ; j < _nFunctions ; j++) {
        ARE::Function *f = _Functions[j] ;
        if (NULL == f) return ERRORCODE_fn_ptr_is_NULL ;
        ARE::FunctionNN *fNN_in = dynamic_cast<ARE::FunctionNN*>(f) ;
        if (nullptr != fNN_in){
            ++n_fNN_in ;
        }
        if (0 == f->N()) bews->ApplyFnCombinationOperator(const_factor, f->ConstValue()) ;
        else {
            flist.push_back(f) ;
            f->ComputeArgumentsPermutationList(w, vars); } // ASK?????  what if f is a NN?!
    }

    float zero = 0.1*pow(10,-34);

    for ( i = 0 ; i < total_samples ; ++i) {
        // generate assignment to fNN arguments
        for (int32_t j = 0 ; j < fNN->N() ; ++j) {
            int32_t v = fNN->Argument(j) ;
            int32_t domain_size_of_v = problem->K(v) ;
            int32_t value = RNG.randInt(domain_size_of_v-1) ;
            vals[j] = value ;
        }

        // enumerate all current variable values; compute bucket value for each configuration and combine them using elimination operator...
        ARE_Function_TableType V = bews->VarEliminationDefaultValue() ;
        for (int32_t j = 0 ; j < k ; ++j) {
            vals[fNN->N()] = j ;
            ARE_Function_TableType v = bews->FnCombinationNeutralValue() ;
            // compute value for this configuration : fNN argument assignment + _V=j
            for (int32_t l = 0 ; l < _nFunctions ; ++l) {
                ARE::Function *f = _Functions[l];
                if (NULL == f) continue ;
                double fn_v = f->TableEntryEx(vals, problem->K()); //This would make us problem specifially when privious ones be NN
                bews->ApplyFnCombinationOperator(v, fn_v) ;
            }
            ApplyVarEliminationOperator(varElimOperator, problem->FunctionsAreConvertedToLogScale(), V, v) ;
        }

        bews->ApplyFnCombinationOperator(V, const_factor) ;

        if (i < n_train_samples) {
            for (int32_t m = 0; m < fNN->N(); ++m) {
                if (convert_exp)
                {
                    int32_t v = fNN->Argument(m);
                    int32_t domain_size_of_v = problem->K(v) - 1;
                    train_samples[i][m] = (float) 2*vals[m]/domain_size_of_v -1;
                } else
                    train_samples[i][m] = vals[m];
            }
            if (convert_exp) {
                V = exp(V);
            }
            train_sample_values[i] = V; 

            if (V < fNN->ln_min_value){
                fNN->ln_min_value = V;
            }
            if (V > fNN->ln_max_value)
                fNN->ln_max_value = V;

        } else if (i >= n_train_samples && i < n_train_samples + n_val_samples) {
            for (int32_t m = 0; m < fNN->N(); ++m) {
                if (convert_exp){
                    int32_t v = fNN->Argument(m);
                    int32_t domain_size_of_v = problem->K(v) -1;
                    val_samples[i - n_train_samples][m]  = (float) 2*vals[m]/domain_size_of_v -1;
                } else
                    val_samples[i - n_train_samples][m] = vals[m];
            }
            if (convert_exp) {
                V = exp(V);
            }
            val_sample_values[i - n_train_samples] = V; 

        } else if (i >= n_train_samples + n_val_samples && i < n_train_samples + n_val_samples + n_test_samples) {
            for (int32_t m = 0; m < fNN->N(); ++m) {
                if (convert_exp){
                    int32_t v = fNN->Argument(m);
                    int32_t domain_size_of_v = problem->K(v) -1;
                    test_samples[i - n_train_samples - n_val_samples][m] = (float) 2*vals[m]/domain_size_of_v -1;
                }else
                    test_samples[i - n_train_samples - n_val_samples][m] = vals[m];
            }
            if (convert_exp) {
                V = exp(V);
            }
            test_sample_values[i - n_train_samples - n_val_samples] = V; //should be V
        }
    }

    std::string o_file = global_config.out_file + "bucket-training.txt";
    std::ofstream f;
    f.open(o_file,std::ios_base::app);
    printf("sampling finished-----");

    DATA_SAMPLES *DS_train, *DS_val, *DS_test;

    fNN->log_sum_exp(train_sample_values,n_train_samples);

    DS_train = fNN->samples_to_data(train_samples, train_sample_values, fNN->N(), n_train_samples);
    DS_val = fNN->samples_to_data(val_samples, val_sample_values, fNN->N(), n_val_samples);
    DS_test = fNN->samples_to_data(test_samples, test_sample_values, fNN->N(), n_test_samples);

    if (0 == global_config.network.compare("net"))
        fNN->Train(DS_train, DS_val,DS_test,_V);
    else
        fNN->Train_ped(DS_train, DS_val,DS_test,_V);

    if (f.is_open())
    {
        printf("writing to the file %s", o_file.c_str());
        f<< fNN->ln_max_value <<'\t' << fNN->ln_min_value<<'\t'<< n_fNN_in << '\t' << n_train_samples << '\n';
        f.close();
    }

    BucketElimination::Bucket *b_ancestor = _Bucket->ParentBucket();
    std::cout<< b_ancestor->V();

    printf("Bucket number ---- %d", _V);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    double duration = microseconds*1.0 / 1000000;
    this->Workspace()->time_ComputeOutputFunction_NN += duration;
    this->Workspace()->count_ComputeOutputFunction_NN++;
    printf(" ComputeOutputFunction_NN Time: %f Count: %d Average: %f\n", this->Workspace()->time_ComputeOutputFunction_NN, this->Workspace()->count_ComputeOutputFunction_NN, this->Workspace()->time_ComputeOutputFunction_NN/this->Workspace()->count_ComputeOutputFunction_NN);
    printf(" TableEntryEx Time: %f Count: %d Average: %f\n", this->Workspace()->time_TableEntryEx, this->Workspace()->count_TableEntryEx, this->Workspace()->time_TableEntryEx / this->Workspace()->count_TableEntryEx);
    printf(" Train Time: %f Count: %d Average: %f\n", this->Workspace()->time_Train, this->Workspace()->count_Train, this->Workspace()->time_Train / this->Workspace()->count_Train);

    return 0 ;
}





