#include "DeepLAB.h"

using namespace std;
using namespace DL;

Tensor::Tensor( int iDim1, int iDim2, int iDim3) :	
m_pfData(NULL),
m_iRow(iDim1),
m_iColumn(iDim2),
m_iDepth(iDim3),
m_iSizeMatrix(iDim1*iDim2),
m_iVolumCube(iDim1*iDim2*iDim3),
m_iDataLength(iDim1*iDim2*iDim3),
m_iOperationWithNext(NONE),
m_pTNext(NULL),
m_pTBack(NULL),
m_iPropagation(FORWARD_PROPAGATION)
{
	//cout << "constructor" << endl;

	m_pfData = new float[m_iDataLength];
	// delete Tensor::~Tensor()

}

Tensor::Tensor( Tensor& tens ) :
m_iRow(tens.GetRow()),
m_iColumn(tens.GetCol()),
m_iDepth(tens.GetDepth()),
m_iSizeMatrix(tens.GetMatrixSize()),
m_iVolumCube(tens.GetCubeSize()),
m_iDataLength(tens.GetDataLength()),
m_iOperationWithNext(tens.GetOperationWithNext()),
m_pTNext(NULL),
m_pTBack(NULL),
m_iPropagation(FORWARD_PROPAGATION)
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
	return m_pfData[iRow + iColumn * m_iColumn + iDepth * m_iSizeMatrix];
}


void Tensor::SetData( float* a_fArr )
{
	int i = GetDataLength();
	while ( i-- ) 
	{
		m_pfData[i] = a_fArr[i];
	}
}


void Tensor::SetZero()
{
	int i = GetDataLength();
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



void Tensor::tr()
{
	int iTempRow;
	iTempRow = m_iRow;
	m_iRow = m_iColumn;
	m_iColumn = iTempRow;
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
	return 0;
}
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



Tensor	Sigmoid( Tensor& tens )
{
	
}

Tensor& Tensor::operator = ( Tensor& tens )
{
	int i = GetDataLength();
	
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
	int i = GetDataLength();
	while ( i-- ) 
	{
		m_pfData[i] = a_fArr[i];
	}
	return *this;
}

Tensor& Tensor::operator = ( int* a_iArr )
{
	int i = GetDataLength();
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
	int row1 = GetRow(); 
	int col1 = GetCol(); 
	int dep = GetDepth();
	
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

Tensor Tensor::operator * ( float fVal )
{
	int i = GetDataLength();
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = m_pfData[i] * fVal;
	}

	return Result;
}

Tensor Tensor::operator * ( int iVal )
{
	int i = GetDataLength();
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = m_pfData[i] * iVal;
	}
	
	return Result;
}

Tensor Tensor::operator + ( Tensor& tens )
{
	int i = GetDataLength();
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = m_pfData[i] + tens[i];
	}

	return Result;
}

Tensor Tensor::operator + ( float fVal )
{
	int i = GetDataLength();
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = m_pfData[i] + fVal;
	}
	
	return Result;
}

Tensor Tensor::operator + ( int iVal )
{
	int i = GetDataLength();
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = m_pfData[i] + iVal;
	}
	
	return Result;
}

Tensor	Tensor::operator - ( Tensor& tens )
{
	int i = GetDataLength();
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = m_pfData[i] - tens[i];
	}

	return Result;
}

Tensor	Tensor::operator - ( float fVal )
{
	int i = GetDataLength();
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = m_pfData[i] - fVal;
	}
	
	return Result;
}

Tensor	Tensor::operator - ( int iVal )
{
	int i = GetDataLength();
	Tensor Result(m_iRow, m_iColumn );
	while( i-- )
	{
		Result[i] = m_pfData[i] - iVal;
	}
	
	return Result;
}


Tensor Tensor::operator / ( float fVal )
{
	int i = GetDataLength();
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
	int i = GetDataLength();
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