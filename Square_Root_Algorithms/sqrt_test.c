//#include "config.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <stdbool.h>
#include <unistd.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <getopt.h>
#include <limits.h>
#define SQUARE_ROOT_OF_2 1.41421356237

float* createTable() {

    float* table = malloc(16 * sizeof(float));
    if(table == NULL){
        printf("Table can not be allocated.\n");
        abort();
    }

// MANTISSA VALUES FOR THE FIRST 4 BITS, MANTISSA'S BINARY REPRESANTATION AND SQUARE ROOT IN DECIMAL
// BINARY REPRESENTATION IS THE ARRAY INDEX OF THE SQUARE ROOT OF THAT MANTISSA AT THE SAME TIME
//              DECIMAL SQRT         BINARY      DECIMAL
    *table = 1.0;                    //0000      1.0000
    *(table+1) =1.0307764064;        //0001      1.0625 
    *(table+2) =1.06066017178;       //0010      1.1250
    *(table+3) =1.08972473589;      //0011      1.1875
    *(table+4) =1.11803398875;      //0100      1.2500
    *(table+5) =1.14564392374;      //0101      1.3125
    *(table+6) =1.17260393996;      //0110      1.3750
    *(table+7) =1.19895788083;      //0111      1.4375
    *(table+8) =1.22474487139;      //1000      1.5000
    *(table+9) =1.25;               //1001      1.5625
    *(table+10) =1.2747548784;       //1010      1.6250
    *(table+11) =1.29903810568;      //1011      1.6875
    *(table+12) =1.32287565553;      //1100      1.7500
    *(table+13) =1.34629120178;      //1101      1.8125
    *(table+14) =1.36930639376;      //1110      1.8750
    *(table+15) =1.39194109071;      //1111      1.9375

    return table;
}

// Table should be created like above, not without using malloc and not checking if the allocation is successfull or not. 
// However for simplicity reasons only for test cases, an array is created without using malloc. Table method is the chosen
// method for standart deviation, but if it were to chosen, the array would be created like above, not below.


const float table[16] = 
// MANTISSA VALUES FOR THE FIRST 4 BITS, MANTISSA'S BINARY REPRESANTATION AND SQUARE ROOT IN DECIMAL
// BINARY REPRESENTATION IS THE ARRAY INDEX OF THE SQUARE ROOT OF THAT MANTISSA AT THE SAME TIME
//  DECIMAL SQRT        BINARY      DECIMAL
{   1.0,                //0000      1.0000
    1.0307764064,       //0001      1.0625 
    1.06066017178,      //0010      1.1250
    1.08972473589,      //0011      1.1875
    1.11803398875,      //0100      1.2500
    1.14564392374,      //0101      1.3125
    1.17260393996,      //0110      1.3750
    1.19895788083,      //0111      1.4375
    1.22474487139,      //1000      1.5000
    1.25,               //1001      1.5625
    1.2747548784,       //1010      1.6250
    1.29903810568,      //1011      1.6875
    1.32287565553,      //1100      1.7500
    1.34629120178,      //1101      1.8125
    1.36930639376,      //1110      1.8750
    1.39194109071       //1111      1.9375
    };

float approxOne(float x) {
    union { float f; uint32_t i; } u = {x};	

    if(u.f >= 2.0) {
         if ((u.i & 0b00000000100000000000000000000000) == 0b00000000100000000000000000000000) {
        //exponent even
        u.i >>= 1; // shift so that exponent power will be squared
        u.i |= 0b01000000000000000000000000000000; // 0b01000000010000000000000000000000
        u.i &= 0b11011111101111111111111111111111;
    } else {
         //exponent odd
        u.i >>= 1; // shift so that exponent power will be squared
        u.i |= 0b01000000010000000000000000000000; // 0b01000000000000000000000000000000
        u.i &= 0b11011111111111111111111111111111; // 0b11011111011111111111111111111111;
        u.f /= 2;
        //or = 0b01000000010000000000000000000000
    }
    return u.f;
    } else  {
        if ((u.i & 0b00000000100000000000000000000000) == 0b00000000100000000000000000000000) {
        //exponent even
        u.i >>= 1; // shift so that exponent power will be squared
        u.i |= 0b00100000000000000000000000000000; // 
        u.i &= 0b11111111101111111111111111111111;
    } else {
        //exponent odd
        u.i >>= 1; // shift so that exponent power will be squared
        u.i |= 0b00100000010000000000000000000000;
        u.i &= 0b11111111111111111111111111111111;
        u.f /= 2;
    }
	return u.f;
    }
}

//https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Approximations_that_depend_on_the_floating_point_representation
float approxTwo(float z)
{
	union { float f; uint32_t i; } val = {z};
	val.i -= 1 << 23;	
	val.i >>= 1;		
	val.i += 1 << 29;	
	return val.f;		
}

float approxThree(float z)
{
	union { float f; uint32_t i; } val = {z};
	val.i -= 0b00000000100000000000000000000000;	
	val.i >>= 1;		
	val.i += 0b00100000000000000000000000000000;
	return val.f;		
}

//approx1 + 1 Newton
float newton1(float x){
    float y = approxOne(x);
    y = (y + x / y) / 2.0;
    return y;
}

//approx2 + 2 Newton
float newton2(float x){
    float y = approxOne(x);
    y = (y + x / y) / 2.0;
    y = (y + x / y) / 2.0;
    return y;
}

//approx2 + 3 Newton
float newton3(float x){
    float y = approxOne(x);
    y = (y + x / y) / 2.0;
    y = (y + x / y) / 2.0;
    y = (y + x / y) / 2.0; 
    return y;
}

//approx2 + 4 Newton
float newton4(float x){
    float y = approxOne(x);
    y = (y + x / y) / 2.0;
    y = (y + x / y) / 2.0;
    y = (y + x / y) / 2.0;
    y = (y + x / y) / 2.0; 
    return y;
}

//approx1 + 1 Halley
float halley1(float x){
    float g = approxOne(x);
    g = ((g*g*g) + (3*x*g)) / ((3*g*g) + x);
    return g;
}

//approx2 + 2 Halley
float halley2(float x){
    float g = approxOne(x);
    g = ((g*g*g) + (3*x*g)) / ((3*g*g) + x);
    g = ((g*g*g) + (3*x*g)) / ((3*g*g) + x);
    return g;
}

//table with 4 elements
float sqrtTable4(float x) {
    union { float f; uint32_t i; } u = {x};
    int a = (u.i & 0b00000000011111111111111111111111) >> 21; // extract mantissa
    int tester = u.i & 0b00000000100000000000000000000000;
    //only needed for odd, that division
    u.i -= 0b00000000100000000000000000000000; // so that odd powers will be rounded down (5/2 = 2) and mantissa will be increased and even powers shif wont increase mantissa
    u.i >>= 1;
    u.i += 0b00100000000000000000000000000000; // for bias in exponent
    //a = (u.i & 0b00000000011111111111111111111111) >> 19; // extract mantissa
    u.i &= 0b11111111100000000000000000000000; // get rid of mantissa 
    float result = u.f * table[a];
    if (tester == 0) { // exponent is odd
        result *= SQUARE_ROOT_OF_2;
    }
    return result;
}

//table with 8 elements
float sqrtTable8(float x) {
    union { float f; uint32_t i; } u = {x};
    int a = (u.i & 0b00000000011111111111111111111111) >> 20; // extract mantissa
    int tester = u.i & 0b00000000100000000000000000000000;
    //only needed for odd, that division
    u.i -= 0b00000000100000000000000000000000; // so that odd powers will be rounded down (5/2 = 2) and mantissa will be increased and even powers shif wont increase mantissa
    u.i >>= 1;
    u.i += 0b00100000000000000000000000000000; // for bias in exponent
    //a = (u.i & 0b00000000011111111111111111111111) >> 19; // extract mantissa
    u.i &= 0b11111111100000000000000000000000; // get rid of mantissa 
    float result = u.f * table[a];
    if (tester == 0) { // exponent is odd
        result *= SQUARE_ROOT_OF_2;
    }
    return result;
}

float sqrtTable16(float x) {
    union { float f; uint32_t i; } u = {x};
    int a = (u.i & 0b00000000011111111111111111111111) >> 19; // extract mantissa
    int tester = u.i & 0b00000000100000000000000000000000;
    //only needed for odd, that division
    u.i -= 0b00000000100000000000000000000000; // so that odd powers will be rounded down (5/2 = 2) and mantissa will be increased and even powers shif wont increase mantissa
    u.i >>= 1;
    u.i += 0b00100000000000000000000000000000; // for bias in exponent
    //a = (u.i & 0b00000000011111111111111111111111) >> 19; // extract mantissa
    u.i &= 0b11111111100000000000000000000000; // get rid of mantissa 
    float result = u.f * table[a];
    if (tester == 0) { // exponent is odd
        result *= SQUARE_ROOT_OF_2;
    }
    return result;
}

//Normal
float Q_rsqrt( float number )
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;                       // evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the fuck? 
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
	//y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y;
}

//Normal returning square root
float Q_sqrt_1( float number )
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;                       // evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the fuck? 
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
	//y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y * number;
}

//Comment out deleted + returning square root
float Q_sqrt_2( float number )
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;                       // evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the fuck? 
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y * number;
}

//Q_rsqrt normal, returning square root + 1 Newton
float newtonQsqrt(float x){
    float y = Q_sqrt_1(x);
    y = (y + x / y) / 2.0;
    return y;
}

//Q_rsqrt + Goldschmidt
float goldschmidtsSqrt2Iteration(float x) {
   float y = Q_rsqrt(x);
   
   float g = x * y;
   float h = y / 2;
   float r = 0.5 - g*h;

   g = g + g*r;
   h = h + h*r;
   r = 0.5 - g*h;
   
   return g;
}

double get_sqrt_time ( float (*f)(float)) {

    float result;
    double time = 0; 
    struct timespec start;      
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0.0; i < INT_MAX;i++){
        result = (*f)(i);
    }
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    time += end.tv_sec - start.tv_sec + 1e-9 * (end.tv_nsec - start.tv_nsec);

    sleep(2);

    int x = 0;
    // to minimise cache affects, do something
    for (int i = INT_MAX; i > 30; i--){
        x++;
        x--;
    }

    sleep(1);
    
    return time;
}

void get_error_rate ( float (*f)(float)) {
    float max = 0.00000005;
    float min = 1.0;
    float error = 0.0;
    float checker;
    float x = 100.0;
    float totalError = 0.0;
    for (int i = 1; i < INT_MAX; i++) {
        checker = (*f)(i) - sqrt(i);
        if (checker < 0.0){
            checker *= -1.0;
            }
        error = (checker / sqrt(i)) * x;
        if (max < error)
        max = error;
        if (min > error)
        min = error;
        totalError += error;
        }
        float divisor = INT_MAX;
        float percentage = totalError / divisor;
        printf("Error percentage:   %.15f\n", percentage);
        printf("Max:                %.15f\n", max);
        printf("Min:                %.15f\n", min);
        printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
}

void printAllErrorRates(){
     // Error Rate Calculations:
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("ERROR RATE/PERCENTAGE CALCULATIONS:\n");
    printf("Average, max and min error rates are calculated\n");
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");

    printf("Function: approxOne\n");
    get_error_rate(&approxOne);

    printf("Function: approxTwo\n");
    get_error_rate(&approxTwo);

    printf("Function: approx + 1Newton\n");
    get_error_rate(&newton1);

    printf("Function: approx + 2Newton\n");
    get_error_rate(&newton2);

    printf("Function: approx + 3Newton\n");
    get_error_rate(&newton3);

    printf("Function: approx + 4Newton\n");
    get_error_rate(&newton4);

    printf("Function: approx + 1Halley\n");
    get_error_rate(&halley1);

    printf("Function: approx + 2Halley\n");
    get_error_rate(&halley2);

    printf("Function: Table 4 Element\n");
    get_error_rate(&sqrtTable4);

    printf("Function: Table 8 Element\n");
    get_error_rate(&sqrtTable8);

    printf("Function: Table 16 Element\n");
    get_error_rate(&sqrtTable16);

    printf("Function: Goldschmidt 2 Iteration(\n");
    get_error_rate(&goldschmidtsSqrt2Iteration);

    printf("Function: Q_sqrt_1\n");
    get_error_rate(&Q_sqrt_1);

    printf("Function: Q_sqrt_2\n");
    get_error_rate(&Q_sqrt_2);

    printf("Function: Q_sqrt + newton\n");
    get_error_rate(&newtonQsqrt);

    printf("Error rate/percentage calculations are finished\n");
    				printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    }

void printAllRunTimes(){
    // Time calculations:
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("TIME CALCULATIONS:\n");
    printf("All of the functions are 5 times executed, and total calculated time is divided to 5.0.\n");
    printf("Between every execution, a for loop is executed to minimise the cache affects.\n");
    printf("Between every tests there is a 1 second break.\n");
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    
    // library sqrt causes compiler issues on locak computer, so had to be done like that.
    double total_time = 0.0;
        for (int i = 0; i < 5; i++){
    float result;
    double time = 0; 
    struct timespec start;      
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0.0; i < INT_MAX;i++){
        result = sqrt(i);
    }
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    time += end.tv_sec - start.tv_sec + 1e-9 * (end.tv_nsec - start.tv_nsec);

    int x = 0;
    // to minimise cache affects, do something
    for (int i = INT_MAX; i > 30; i--){
        x++;
        x--;
    }

    sleep(1);

    total_time += time;
}

    // Function Name: sqrt
    printf("sqrt time is:                   %.15f\n", (total_time / 5.0));
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");

    // Function Name: T16
    printf("Table 16 Elements time is:      %.15f\n", (get_sqrt_time(&sqrtTable16) + get_sqrt_time(&sqrtTable16) + get_sqrt_time(&sqrtTable16) + get_sqrt_time(&sqrtTable16) + get_sqrt_time(&sqrtTable16)) / 5.0);
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    // Function Name: A1 + 1N
    printf("A1 + 1Newton time is:           %.15f\n", (get_sqrt_time(&newton1) + get_sqrt_time(&newton1) + get_sqrt_time(&newton1) + get_sqrt_time(&newton1) + get_sqrt_time(&newton1)) / 5.0);
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    // Function Name: T4
    printf("Table 4 Element time is:        %.15f\n", (get_sqrt_time(&sqrtTable4) + get_sqrt_time(&sqrtTable4) + get_sqrt_time(&sqrtTable4) + get_sqrt_time(&sqrtTable4) + get_sqrt_time(&sqrtTable4)) / 5.0);
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    // Function Name: approxOne
    printf("approxOne time is:              %.15f\n", (get_sqrt_time(&approxOne) + get_sqrt_time(&approxOne) + get_sqrt_time(&approxOne) + get_sqrt_time(&approxOne) + get_sqrt_time(&approxOne)) / 5.0);
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    // Function Name: T8
    printf("Table 8 Element time is:        %.15f\n", (get_sqrt_time(&sqrtTable8) + get_sqrt_time(&sqrtTable8) + get_sqrt_time(&sqrtTable8) + get_sqrt_time(&sqrtTable8) + get_sqrt_time(&sqrtTable8)) / 5.0);
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    // Function Name: approxTwo
    printf("approxTwo time is:              %.15f\n", (get_sqrt_time(&approxTwo) + get_sqrt_time(&approxTwo) + get_sqrt_time(&approxTwo) + get_sqrt_time(&approxTwo) + get_sqrt_time(&approxTwo)) / 5.0);
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    // Function Name: A1 + 2N
    printf("A1 + 2Newton time is:           %.15f\n", (get_sqrt_time(&newton2) + get_sqrt_time(&newton2) + get_sqrt_time(&newton2) + get_sqrt_time(&newton2) + get_sqrt_time(&newton2)) / 5.0);
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    // Function Name: A1 + 1H
    printf("A1 + 1Halley time is:           %.15f\n", (get_sqrt_time(&halley1) + get_sqrt_time(&halley1) + get_sqrt_time(&halley1) + get_sqrt_time(&halley1) + get_sqrt_time(&halley1)) / 5.0);
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    // Function Name: A1 + 3N
    printf("A1 + 3Newton time is:           %.15f\n", (get_sqrt_time(&newton3) + get_sqrt_time(&newton3) + get_sqrt_time(&newton3) + get_sqrt_time(&newton3) + get_sqrt_time(&newton3)) / 5.0);
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    // Function Name: A1 + 2H
    printf("A1 + 2Halley time is:            %.15f\n", (get_sqrt_time(&halley2) + get_sqrt_time(&halley2) + get_sqrt_time(&halley2) + get_sqrt_time(&halley2) + get_sqrt_time(&halley2)) / 5.0);
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    // Function Name: newtonQsqrt
    //37.664498237600000
    printf("newtonQsqrt time is:            %.15f\n", (get_sqrt_time(&newtonQsqrt) + get_sqrt_time(&newtonQsqrt) + get_sqrt_time(&newtonQsqrt) + get_sqrt_time(&newtonQsqrt) + get_sqrt_time(&newtonQsqrt)) / 5.0);
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    // Function Name: Q_sqrt_1
    // 27.798328898000001
    printf("Q_sqrt_1 time is:               %.15f\n", (get_sqrt_time(&Q_sqrt_1) + get_sqrt_time(&Q_sqrt_1) + get_sqrt_time(&Q_sqrt_1) + get_sqrt_time(&Q_sqrt_1) + get_sqrt_time(&Q_sqrt_1)) / 5.0);
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    // Function Name: Q_sqrt_2
    // 35.197034343400006
    printf("Q_sqrt_2 time is:               %.15f\n", (get_sqrt_time(&Q_sqrt_2) + get_sqrt_time(&Q_sqrt_2) + get_sqrt_time(&Q_sqrt_2) + get_sqrt_time(&Q_sqrt_2) + get_sqrt_time(&Q_sqrt_2)) / 5.0);
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    // Function Name: Goldschmidt 2 Iteration
    // 35.496433105400001
    printf("goldschmidt2 time is:           %.15f\n", (get_sqrt_time(&goldschmidtsSqrt2Iteration) + get_sqrt_time(&goldschmidtsSqrt2Iteration) + get_sqrt_time(&goldschmidtsSqrt2Iteration) + get_sqrt_time(&goldschmidtsSqrt2Iteration) + get_sqrt_time(&goldschmidtsSqrt2Iteration)) / 5.0);
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    // Function Name: approxThree
    printf("approxThree time is:            %.15f\n", (get_sqrt_time(&approxThree) + get_sqrt_time(&approxThree) + get_sqrt_time(&approxThree) + get_sqrt_time(&approxThree) + get_sqrt_time(&approxThree)) / 5.0);
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("Time calculations are finished\n");
    printf("-------------------------------------------------------------------------------------------------------------------------------------------------------\n");
}

int main() {
    printAllErrorRates();
    //printAllRunTimes();
    return 0;
}
