#include <iostream>
#include <math.h>

#define		e		2.7182818284

namespace DL
{
	enum		// status constants
	{
		NONE,

		// Operation Label
		TRANSPOSE,
		DETERMINANT,
		INVERSE,
		IDENTITY,

		MATMUL,
		ELEMENTMUL,
		CONV,

		// Activation Label
		SIGMOID,
		RELU,

	};


	class Tensor
	{
	public: // Functions 
		Tensor( int iDim1 = 1, int iDim2 = 1, float* fArr = NULL, int iDim3 = 1 );
		Tensor( Tensor& tens );
		~Tensor();

	public:
		// access functions
		float&	at( int iRow = 0, int iCol = 0, int iDepth = 0 );
		void	SetData( float* a_fArr );
		void	SetRow( float* a_fArr, int iRow = 0 );
		void	SetCol( float* a_fArr, int iCol = 0 );
		void	SetLinkLabel( int iLabel )					{ m_iLinkLabel = iLabel;		}
		void	SetLayerLabel( int iLabel )					{ m_iLayerLabel = iLabel;		}
		void	SetNextLink( Tensor* tens )					{ m_pTNext = tens;				}
		void	SetBackLink( Tensor* tens )					{ m_pTBack = tens;				}
		void	SetFixLabel( bool bFixLabel )				{ m_bFixLabel = bFixLabel;		}

		void	SetZero();

		const int		GetDataLength()			const		{ return m_iDataLength;			}
		const int		GetRow()				const		{ return m_iRow;				}
		const int		GetCol()				const 		{ return m_iColumn;				}
		const int		GetDepth()				const		{ return m_iDepth;				}
		const int		GetMatrixSize()			const		{ return m_iSizeMatrix;			}
		const int		GetCubeSize()			const		{ return m_iVolumCube;			}
		const int		GetLinkLabel()			const		{ return m_iLinkLabel;			}
		const int		GetLayerLabel()			const		{ return m_iLayerLabel;			}
 		Tensor*			GetNextLink()			const		{ return m_pTNext;				}
  		Tensor*			GetBackLink()			const		{ return m_pTBack;				}
		const bool		GetFixLabel()			const		{ return m_bFixLabel;			}
		
		Tensor&	GetRow( int iIndex )	const;
		


		// matrix functions
		Tensor	tr();
		Tensor	inv();
		void	I();
		float	det();
		Tensor	Sigmoid();

		Tensor	ElemMul(Tensor& tens);
		void	print( Tensor* tens = NULL );
		void	InitRand();
		Tensor	ReduceSize();

		
		
		// learning algorithms
		void ForwardPropagaion();
		void BackPropagaion( Tensor& tens, float fLearningRate );
		

		

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
		Tensor	operator - ();

		Tensor	operator * ( Tensor&	tens	);
		Tensor	operator * ( float		fVal	);
		Tensor	operator * ( int		iVal	);

		Tensor	operator / ( float		fVal	);
		Tensor	operator / ( int		iVal	);

		float&	operator [] ( int iIndex );	

		Tensor& operator >> ( Tensor&	tens	);


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
		int		m_iLinkLabel;
		int		m_iLayerLabel;
		bool	m_bFixLabel;

		// link
		Tensor* m_pTNext;
		Tensor* m_pTBack;
	};

	
}


