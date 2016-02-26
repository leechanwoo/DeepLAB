#include <iostream>
#include <math.h>

namespace DL
{
	enum
	{
		NONE = 0,
		MATMUL,
		CONV,
		ACTIVATE
	};



	class Tensor
	{
	public: // Functions 
		Tensor( int iDim1 = 1, int iDim2 = 1, int iDim3 = 1 );
		Tensor( Tensor& tens );
		~Tensor();

	public:
		// access functions
		float&	at( int iRow = 0, int iCol = 0, int iDepth = 0 );
		void	SetData( float* a_fArr );
		void	SetRow( float* a_fArr, int iRow = 0 );
		void	SetCol( float* a_fArr, int iCol = 0 );
// 		void	SetRow( Tensor& tensor, int iRow = 0 );
// 		void	SetCol( Tensor& tensor, int iCol = 0 );
		void	SetNextLink( Tensor* tens )					{ m_pTNext = tens;				}
		void	SetBackLink( Tensor* tens )					{ m_pTBack = tens;				}

		void	SetZero();

		const int		GetDataLength()			const		{ return m_iDataLength;			}
		const int		GetRow()				const		{ return m_iRow;				}
		const int		GetCol()				const 		{ return m_iColumn;				}
		const int		GetDepth()				const		{ return m_iDepth;				}
		const int		GetMatrixSize()			const		{ return m_iSizeMatrix;			}
		const int		GetCubeSize()			const		{ return m_iVolumCube;			}
		const int		GetOperationWithNext()	const		{ return m_iOperationWithNext;	}
		const int		GetPropagationMode()	const		{ return m_iPropagation;		}
 		Tensor*			GetNextLink()			const		{ return m_pTNext;				}
 		Tensor*			GetBackLink()			const		{ return m_pTBack;				}
		
		Tensor&	GetRow( int iIndex )	const;

		// matrix functions
		void	tr();
		Tensor	inv();
		void	I();
		float	det();
		Tensor	Sigmoid( Tensor& tens );
		
		
		
		// learning algorithms
		void ForwardPropagaion();
		void BackParopagaion();
		

		

	public: // operator
		Tensor& operator = ( Tensor&	tens	);
		Tensor& operator = ( float*		a_fArr	);
		Tensor& operator = ( int*		a_iArr	);
		Tensor& operator = ( float		fVal	);
		Tensor& operator = ( int		iVal	);

		Tensor  operator + ( Tensor&	tens	);
		Tensor	operator + ( float		fVal	);
		Tensor  operator + ( int		iVal	); 

		Tensor	operator - ( Tensor&	tens	);
		Tensor  operator - ( float		fVal	);
		Tensor  operator - ( int		iVal	);

		Tensor	operator * ( Tensor&	tens	);
		Tensor	operator * ( float		fVal	);
		Tensor	operator * ( int		iVal	);

		Tensor	operator / ( float		fVal	);
		Tensor	operator / ( int		iVal	);

		float&	operator [] ( int iIndex );	

		Tensor&	operator >> ( Tensor&	tens	);

		

	public: // Variables


	protected: // Functions


	protected: // Variables

		// Data
		float*	m_pfData;
		int		m_iDataLength;
		
		// Dimension
		int		m_iRow;
		int		m_iColumn;
		int		m_iDepth;
		int		m_iSizeMatrix;
		int		m_iVolumCube;

		// operator memory
		int		m_iOperationWithNext;
		int		m_iPropagation;

		// link
		Tensor* m_pTNext;
		Tensor* m_pTBack;
	};



}