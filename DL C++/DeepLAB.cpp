#include "DeepLAB.h"

using namespace std;
using namespace DL;

Tensor::Tensor( int iDim1, int iDim2, float* fArr, int iDim3 ) :	
m_pfData(NULL),
m_iRow(iDim1),
m_iColumn(iDim2),
m_iDepth(iDim3),
m_iSizeMatrix(iDim1*iDim2),
m_iVolumCube(iDim1*iDim2*iDim3),
m_iDataLength(iDim1*iDim2*iDim3),
m_pTNext(NULL),
m_pTBack(NULL)
{
	//cout << "constructor" << endl;

	m_pfData = new float[m_iDataLength];
	// delete Tensor::~Tensor()

	int i = m_iDataLength;
	if ( fArr != NULL )
	{
		while( i-- )
		{
			m_pfData[i] = fArr[i];
		}
	}
	


}

Tensor::Tensor( Tensor& tens ) :
m_iRow(tens.GetRow()),
m_iColumn(tens.GetCol()),
m_iDepth(tens.GetDepth()),
m_iSizeMatrix(tens.GetMatrixSize()),
m_iVolumCube(tens.GetCubeSize()),
m_iDataLength(tens.GetDataLength()),
m_pTNext(NULL),
m_pTBack(NULL)
{
	int i = tens.GetDataLength();

	m_pfData = new float[i];
	while( i-- )
	{
		m_pfData[i] = tens[i];
	}
}



Tensor::~Tensor()
{
	if ( m_pfData != NULL )
	{
		delete[] m_pfData;
	}

}



float& Tensor::at( int iRow, int iColumn, int iDepth )
{
	return m_pfData[iRow + iColumn * m_iRow + iDepth * m_iSizeMatrix];
}


void Tensor::SetData( float* a_fArr )
{
	int i = m_iDataLength;
	while ( i-- ) 
	{
		m_pfData[i] = a_fArr[i];
	}
}


void Tensor::SetZero()
{
	int i = m_iDataLength;
	while ( i-- ) 
	{
		m_pfData[i] = 0;
	}
}

void Tensor::SetRow( float* a_fArr, int iRow )
{
	int i = GetCol();
	while( i-- )
	{
		at(iRow, i) = a_fArr[i];
	}
}	


void Tensor::SetCol( float* a_fArr, int iCol )
{
	int i = GetRow();
	while( i-- )
	{
		at(i, iCol) = a_fArr[i];
	}
}



Tensor Tensor::tr()
{
	Tensor Result( m_iColumn, m_iRow );
	int i = m_iDataLength;
	while( i-- )
	{
		Result[i] = m_pfData[i];
	}

	return Result;	
}

float Tensor::det()
{
switch( m_iRow )
{

case 0: 
	return 0;
case 1:
	return m_pfData[0];
case 2:
	return at(0,0)*at(1,1) - at(0,1)*at(1,0);	
case 3:
	return 
		at(0,0)*( at(1,1)*at(2,2)-at(1,2)*at(2,1) )
		-at(0,1)*( at(1,0)*at(2,2)-at(1,2)*at(2,0) )
		+at(0,2)*( at(1,0)*at(2,1)-at(1,1)*at(2,0) ); 
case 4:
	return 
		at(0,0)* ( at(1,1) * ( at(2,2)*at(3,3)-at(2,3)*at(3,2) )
					-at(1,2) * ( at(2,1)*at(3,3)-at(2,3)*at(3,1) )
					+at(1,3) * ( at(2,1)*at(3,2)-at(2,2)*at(3,1) ))
		-at(0,1)* ( at(1,0) * ( at(2,2)*at(3,3)-at(2,3)*at(3,2) )
					-at(1,2) * ( at(2,0)*at(3,3)-at(2,3)*at(3,0) )
					+at(1,3) * ( at(2,0)*at(3,2)-at(2,2)*at(3,0) ))
		+at(0,2)* ( at(1,0) * (at(2,1)*at(3,3)-at(2,3)*at(3,1) )
					-at(1,1) * ( at(2,0)*at(3,3)-at(2,3)*at(3,0) )
					+at(1,3) * ( at(2,0)*at(3,1)-at(2,1)*at(3,0) ))
		-at(0,3)* ( at(1,0) * (at(2,1)*at(3,2)-at(2,2)*at(3,1) )
					-at(1,1) * (at(2,0)*at(3,2)-at(2,2)*at(3,0) )
					+at(1,2) * (at(2,0)*at(3,1)-at(2,1)*at(3,0) ));
					
default:

	float det = 0;
	int p, h, k, i, j;
	int iMatSize = m_iRow;
	Tensor temp(iMatSize,iMatSize);
	Tensor Resized( iMatSize-1, iMatSize-1 );
	
	for (p = 0; p < iMatSize; p++) {
		h = 0;
		k = 0;
		for (i = 1; i < iMatSize; i++) 
		{
			for (j = 0; j < iMatSize; j++) 
			{
				if (j == p) 
				{
					continue;
				}
				temp.at(h,k) = at(i,j);
				k++;

				if (k == iMatSize - 1) {
					h++;
					k = 0;
				}
			}
		}
		Resized = temp.ReduceSize();
		det = det + at(0, p) * (float)pow(-1, p)*Resized.det();
	}
	return det;

}
}


Tensor Tensor::ReduceSize()
{
	Tensor Result( m_iRow-1, m_iColumn-1 );
	
	int i = m_iRow-1;
	int j = m_iColumn-1;
	

	while( j-- )
	{
		while( i-- )
		{
			Result.at(i,j) = at(i,j);
		}
		i = GetRow()-1;
	}

	return Result;

}

void Tensor::I()
{
	int i = GetRow();
	SetZero();
	while( i-- )
	{
		at(i,i) = 1.0f;
	}
}

Tensor Tensor::inv()
{
	float fDet = det();
	float fDet_ = 0;
	
	fDet = 1.0f/fDet;
	
	int i, j, k;
	i = j = k = GetRow();
	
	float* fVec = new float[i];
	Tensor IdentityMat(i,i);
	Tensor Result(i,i);
	Result.SetZero();
	IdentityMat.I();
	
	
	while ( i-- )
	{	
		while ( j-- )
		{
			while( k-- )
			{
				fVec[k] = at(k,j);
				at(k,j) = IdentityMat.at(k,i);		
			}
			Result.at(j,i) = det()*fDet;
			k = GetRow();

			while( k-- )
			{
				at(k,j) = fVec[k];
			}
			k = GetRow();
		}
		j = GetRow();
	}

	delete[] fVec;

	return Result;
}



Tensor Tensor::Sigmoid()
{
	int i = GetRow();
	Tensor Result(i, 1);
	while( i-- )
	{
		Result.at(i) = 1 / (1 + pow(e, -at(i)));
	}

	return Result;
}

Tensor& Tensor::operator = ( Tensor& tens )
{
	if ( GetRow() != tens.GetRow() || GetCol() != tens.GetCol() )
	{
		if ( m_pfData != NULL )
		{
			delete[] m_pfData;
			m_pfData = new float[tens.GetDataLength()];
		}
	}

	int i = tens.GetDataLength();
	while ( i-- ) 
	{
		m_pfData[i] = tens[i];
	}
	
	m_iRow					= tens.GetRow();
	m_iColumn				= tens.GetCol();
	m_iDepth				= tens.GetDepth();
	m_iSizeMatrix			= tens.GetMatrixSize();
	m_iVolumCube			= tens.GetCubeSize();
	m_iDataLength			= tens.GetDataLength();

	return *this;
}


Tensor& Tensor::operator = ( float* a_fArr )
{
	int i = m_iDataLength;
	while ( i-- ) 
	{
		m_pfData[i] = a_fArr[i];
	}
	return *this;
}

Tensor& Tensor::operator = ( int* a_iArr )
{
	int i = m_iDataLength;
	while ( i-- ) 
	{
		m_pfData[i] = a_iArr[i];
	}
	return *this;

}

Tensor& Tensor::operator = ( float	fVal	)
{
	m_pfData[0] = fVal;
	return *this;
}

Tensor& Tensor::operator = ( int iVal )
{
	m_pfData[0] = iVal;
	return *this;
}


Tensor Tensor::operator * ( Tensor& tens )
{
	int row1 = m_iRow; 
	int col1 = m_iColumn; 
	int dep = m_iDepth;
	
	int row2 = tens.GetRow(); 
	int col2 = tens.GetCol(); 

	Tensor Result( row1, col2 );
	Result.SetZero();

	while( dep-- )
	{
		while ( row1-- )
		{ 
			while( col2-- )
			{ 
				while( row2-- )
				{
					Result.at(row1, col2 ) += at( row1, row2 ) * tens.at( row2, col2 );	
				} 
				row2 = tens.GetRow();
			} 
			col2 = tens.GetCol();
		}
		row1 = GetRow();
	}


	return Result;
}


void Tensor::ForwardPropagaion()
{
	Tensor* Here = this;
	Tensor* Weight = GetNextLink();
	Tensor* Output = Weight->GetNextLink();

	while ( true )
	{

		*Output = ((*Weight) * (*Here)).Sigmoid();
	
		Here = Here->GetNextLink()->GetNextLink();

		if ( Output->GetNextLink() == NULL ) return;


		Weight = Weight->GetNextLink()->GetNextLink();
		Output = Output->GetNextLink()->GetNextLink();
	
	}


}

void Tensor::BackPropagaion( Tensor& tens, float fLearningRate )
{
	Tensor* Here = this;
	Tensor* HereWeight = NULL;
	Tensor* InputWeight = Here->GetBackLink();
	Tensor* Input = InputWeight->GetBackLink();
	Tensor Delta( m_iRow );
	
	
	while( true )
	{	
		if ( Here->GetNextLink() == NULL )
		{
			Delta = (*Here - tens).ElemMul(*Here).ElemMul( -(*Here) + 1 );
			*InputWeight = (*InputWeight) - (Delta * (*Input).tr()) * fLearningRate;
		}
		else
		{
			Delta = ((*HereWeight).tr() * Delta).ElemMul( *Here ).ElemMul( -(*Here) + 1 );
			*InputWeight = *InputWeight - (Delta * (*Input).tr()) * fLearningRate;
		}
		
 		Here = Here->GetBackLink()->GetBackLink();
		HereWeight = Here->GetNextLink();

		if ( Here->GetBackLink() == NULL ) return;


		InputWeight = InputWeight->GetBackLink()->GetBackLink();
		Input = Input->GetBackLink()->GetBackLink();
	}

	
}


Tensor Tensor::operator * ( float fVal )
{
	int i = m_iDataLength;
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = m_pfData[i] * fVal;
	}

	return Result;
}

Tensor Tensor::operator * ( int iVal )
{
	int i = m_iDataLength;
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = m_pfData[i] * iVal;
	}
	
	return Result;
}

Tensor Tensor::operator + ( Tensor& tens )
{
	int i = m_iDataLength;
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = m_pfData[i] + tens[i];
	}

	return Result;
}

Tensor Tensor::operator + ( float fVal )
{
	int i = m_iDataLength;
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = m_pfData[i] + fVal;
	}
	
	return Result;
}

Tensor Tensor::operator + ( int iVal )
{
	int i = m_iDataLength;
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = m_pfData[i] + iVal;
	}
	
	return Result;
}

Tensor	Tensor::operator - ( Tensor& tens )
{
	int i = m_iDataLength;
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = m_pfData[i] - tens[i];
	}

	return Result;
}

Tensor	Tensor::operator - ( float fVal )
{
	int i = m_iDataLength;
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = m_pfData[i] - fVal;
	}
	
	return Result;
}

Tensor Tensor::operator - ( int iVal )
{
	int i = m_iDataLength;
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = m_pfData[i] - iVal;
	}
	
	return Result;
}

Tensor Tensor::operator - ()
{
	int i = m_iDataLength;
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = -m_pfData[i];
	}

	return Result;
}


Tensor Tensor::operator / ( float fVal )
{
	int i = m_iDataLength;
	Tensor Result(m_iRow, m_iColumn );

	fVal = 1.0f/fVal;

	while( i-- )
	{
		Result[i] = m_pfData[i] * fVal;
	}

	return Result;
}

Tensor Tensor::operator / ( int iVal )
{
	int i = m_iDataLength;
	float fVal = 0;
	Tensor Result(m_iRow, m_iColumn );
	
	fVal = 1.0f/(float)iVal;
	
	while( i-- )
	{
		Result[i] = m_pfData[i] * fVal;
	}
	
	return Result;
}

float& Tensor::operator[] ( int iIndex )
{
	return m_pfData[iIndex];
}

Tensor& Tensor::operator >> ( Tensor& tens )
{
	m_pTNext = &tens;
	tens.SetBackLink( this );

	return tens;
}


Tensor Tensor::ElemMul(Tensor& tens)
{
	int i = m_iDataLength;
	Tensor Result(m_iRow, m_iColumn);
	while (i--)
	{
		Result[i] = m_pfData[i] * tens[i];
	}

	return Result;
}

void Tensor::InitRand()
{
	int i = m_iDataLength;
	while( i-- )
	{
		int iRandVal = rand();
		m_pfData[i] = (iRandVal&0x01 ? -1 : 1) * (float)iRandVal/RAND_MAX;
	}

}

void Tensor::print( Tensor* tens )
{
	for ( int row = 0; row < GetRow(); row++ )
	{
		if ( tens == NULL )
		{
			for ( int col = 0; col < GetCol(); col++ )
			{
				printf( "%0.2f ", at( row, col ) );
			}
			
		}
		else
		{
			for ( int col1 = 0; col1 < GetCol(); col1++ )
			{
				printf( "%0.2f ", at( row, col1 ) );
			}
			cout << "| ";
			for ( int col2 = 0; col2 < GetCol(); col2++ )
			{
				printf( "%0.2f ", tens->at( row, col2 ) );
			}
			
		}
		cout << endl;
	}

	cout << endl;
}
