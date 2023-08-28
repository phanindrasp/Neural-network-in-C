#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float sig(float z);
float sigder(float z);


void main()
{

    float C=0;      // Cost function


    printf("Welcome to Phani's neural network to solve a classification problem\n");

    // Read the number of layers in the network
    int Nl;
    printf("Enter the number of layers ");
    scanf("%d",&Nl);

    // Read the number of neurons in each layer
    int *Nn;

    Nn=(int *)malloc((Nl)*sizeof(int));

    for (int i=0;i<Nl;i++)
    {
        printf("Enter number of neurons in layer-%d ",i+1);
        scanf("%d",&Nn[i]);
    }

    // Make neuron arrays for each layer of the network
    float **z;
    float **nrn;
    float **dcnrn;

    z=(float **)malloc((Nl)*sizeof(float *));
    nrn=(float **)malloc((Nl)*sizeof(float *));
    dcnrn=(float **)malloc((Nl)*sizeof(float *));

    for (int i=0;i<Nl;i++)
    {
        z[i]=(float *)malloc((Nn[i])*sizeof(float));
        nrn[i]=(float *)malloc((Nn[i])*sizeof(float));
        dcnrn[i]=(float *)malloc((Nn[i])*sizeof(float));
    }

    // Assign random values to all the neurons
    for (int i=0;i<Nl;i++)
    {
       for (int j=0;j<Nn[i];j++)
       {
         z[i][j]=1;
         nrn[i][j]=1;
         dcnrn[i][j]=1;
       }

    }

    // Make bias arrays between every two layers of the network
    float **bs;
    float **dcbs;

    bs=(float **)malloc((Nl-1)*sizeof(float *));
    dcbs=(float **)malloc((Nl-1)*sizeof(float *));

    for (int i=0;i<Nl-1;i++)
    {
        bs[i]=(float *)malloc((Nn[i+1])*sizeof(float));
        dcbs[i]=(float *)malloc((Nn[i+1])*sizeof(float));
    }

    // Assign random values to all the neurons
    for (int i=0;i<Nl-1;i++)
    {
       for (int j=0;j<Nn[i+1];j++)
       {
         bs[i][j]=(rand()%100)/100.0;
         //bs[i][j]=0;
         //printf("bias %f\n",bs[i][j]);
         dcbs[i][j]=0;
       }

    }
//bs[0][0]=-10;
//bs[0][1]=30;
//bs[1][0]=-30;
    // Make weight matrices between every two layers of the network
    float ***W;
    float ***dcW;

    W=(float ***)malloc((Nl-1)*sizeof(float **));
    dcW=(float ***)malloc((Nl-1)*sizeof(float **));

    for (int i=0;i<Nl-1;i++)
    {

        W[i]=malloc(Nn[i]*sizeof(float *));
        dcW[i]=malloc(Nn[i]*sizeof(float *));

        for (int j=0;j<Nn[i];j++)
        {
            W[i][j]=malloc(Nn[i+1]*sizeof(float));
            dcW[i][j]=malloc(Nn[i+1]*sizeof(float));
        }

    }

    // Assign random values to all the weights
    for (int i=0;i<Nl-1;i++)
    {
       for (int j=0;j<Nn[i];j++)
       {
           for (int k=0;k<Nn[i+1];k++)
           {
               W[i][j][k]=(rand()%50+50)/100.0;
               //W[i][j][k]=0;
               //printf("weight %f\n",W[i][j][k]);
               dcW[i][j][k]=0;
           }

       }

    }
//W[0][0][0]=20;
//W[0][1][0]=20;
//W[0][0][1]=-20;
//W[0][1][1]=-20;
//W[1][0][0]=20;
//W[1][1][0]=20;
    // Training on provided data and back propagation

    float lr=10;        // Learning rate for weight and bias

    // Read the training data

    // Count the amount of data available
    FILE *fptr;
    fptr=fopen("train_data.csv","r");
  /*  char chr;
    int data=0;
    chr = getc(fptr);
    while (chr != EOF)
    {
        if (chr == '\n')
        {
            data = data + 1;
        }
        chr = getc(fptr);
    }

    rewind(fptr); //close file.*/
    int data=4;
    printf("There are %d samples in the training dataset\n", data);


    // Read the data into variables
    float **x;  // Input data
    float **y;  // Output data

    x=(float **)malloc((data)*sizeof(float *));
    y=(float **)malloc((data)*sizeof(float *));

    for (int i=0;i<data;i++)
    {
        x[i]=(float *)malloc((Nn[0])*sizeof(float));
        y[i]=(float *)malloc((Nn[Nl-1])*sizeof(float));
    }

    for(int i=0; i<data;i++)
    {
        for(int j=0; j<Nn[0]+Nn[Nl-1];j++)
         {
             if(j<Nn[0])
             {
                 fscanf(fptr,"%f,",&x[i][j]);
             }
             else
             {
                 fscanf(fptr,"%f,",&y[i][j-Nn[0]]);
             }
         }

    }


   // Perform training using the provided data


   int itr=0;

   while(itr<100)
   {

    // Propagate data forward to get output from the network

    //Initialize the values
    C=0;
    for (int i=0;i<Nn[Nl-1];i++)
    {
       dcnrn[Nl-1][i]=0;
    }


    for (int s=0;s<data;s++)
    {
      for (int i=0;i<Nl;i++)
      {
        for (int j=0;j<Nn[i];j++)
        {

          if(i==0)
          {
              z[i][j]=x[s][j];
              nrn[i][j]=z[i][j];
          }


          else
          {
          z[i][j]=0;

          for (int k=0;k<Nn[i-1];k++)
          {
                  z[i][j]+=W[i-1][k][j]*nrn[i-1][k];
          }

          z[i][j]+=bs[i-1][j];
          nrn[i][j]=sig(z[i][j]);

          if(i==Nl-1)
          {
              C+=pow(nrn[Nl-1][j]-y[s][j],2);
              printf("pred and des of %d are %f %f\n",s,nrn[Nl-1][j],y[s][j]);
              dcnrn[Nl-1][j]=2*(nrn[Nl-1][j]-y[s][j]);

          }

          }

        }
      }



  for (int i=Nl-2;i>=0;i--)
     {
         for (int j=0;j<Nn[i];j++)
         {
             dcnrn[i][j]=0;
         }

     }



    // Calculate the gradient of cost function w.r.t weights and biases


     for (int i=Nl-2;i>=0;i--)
     {
         for (int j=0;j<Nn[i];j++)
         {
             for (int k=0;k<Nn[i+1];k++)
             {
                 dcW[i][j][k]+=nrn[i][j]*sigder(z[i+1][k])*dcnrn[i+1][k];
                 dcbs[i][j]+=sigder(z[i+1][k])*dcnrn[i+1][k];
                 dcnrn[i][j]+=sigder(z[i+1][k])*W[i][j][k]*dcnrn[i+1][k];

             }

         }

     }




    }




     for (int i=Nl-2;i>=0;i--)
     {
         for (int j=0;j<Nn[i];j++)
         {
             for (int k=0;k<Nn[i+1];k++)
             {
                 dcW[i][j][k]=dcW[i][j][k]/data;
                 W[i][j][k]-=dcW[i][j][k]*lr;
                 //printf("W is %f\n",W[i][j][k]);
             }

         }

     }

     for (int i=Nl-2;i>=0;i--)
     {
        for (int k=0;k<Nn[i+1];k++)
        {
                dcbs[i][k]=dcbs[i][k]/data;
                bs[i][k]-=dcbs[i][k]*lr;
              //  printf("dcbs is %f\n",dcbs[i][k]);
        }

     }


    itr+=1;
    printf("Loss function at iteration %d is %f\n",itr,C/data);

    }


    printf("Training done!! :)\n");

      for (int i=Nl-2;i>=0;i--)
     {
         for (int j=0;j<Nn[i];j++)
         {
             for (int k=0;k<Nn[i+1];k++)
             {
                 dcW[i][j][k]=dcW[i][j][k]/data;
                 W[i][j][k]-=dcW[i][j][k]*lr;
                 //printf("W and B of %d %d %d is %f,%f\n",i,j,k,W[i][j][k],bs[i][k]);
             }

         }

     }
    // Predict value

   // Propagate data forward to get output from the network

    printf("Enter your input values separated by enter\n");


/*
    for (int i=1;i<Nl;i++)
    {
        for (int j=0;j<Nn[i];j++)
        {
            for (int k=0;k<Nn[i-1];k++)
            {
                z[i][j]+=W[i-1][k][j]*nrn[i-1][k];
            }

          z[i][j]+=bs[i-1][j];
          nrn[i][j]=sig(z[i][j]);
        }

    }
 */
    for (int i=0;i<Nl;i++)
      {
        for (int j=0;j<Nn[i];j++)
        {

          if(i==0)
          {
               scanf("%f",&nrn[0][j]);
              // nrn[0][j]=sig(z[0][j]);
          }

          else
          {
          z[i][j]=0;

          for (int k=0;k<Nn[i-1];k++)
          {
                  z[i][j]+=W[i-1][k][j]*nrn[i-1][k];
          }

          z[i][j]+=bs[i-1][j];
          nrn[i][j]=sig(z[i][j]);

          if(i==Nl-1)
          {

              printf("Output is is %f\n",nrn[Nl-1][j]);

          }

          }
        }
      }

}

float sig(float z)
{
    float r;
    r=1/(1+exp(-z));
/*
    if(z<=0)
    {
        r=0;
    }
    else
    {
        r=z;
    }*/
    return r;
}

float sigder(float z)
{
    float r;
    r=sig(z)*(1-sig(z));
/*
    if(z<=0)
    {
        r=0;
    }
    else
    {
        r=1;
    }*/
    return r;
}
