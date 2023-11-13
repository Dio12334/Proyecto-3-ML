#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>
#include <iostream>
#include <thread>
#include <cmath>
#include <unordered_map>
#include <random> 
#include <algorithm>
#include <sstream>
#include <fstream>

typedef double (*FunctionType)(double);
typedef std::vector<double> Lista;
typedef std::vector<Lista> Matriz;
typedef std::uint32_t u32;

//Funciones {{{1

std::pair<Matriz, Matriz> loadData(){

	Matriz Y(569, Lista(1));
	Matriz X(569, Lista(30));

	std::string file = "data.csv";
	std::ifstream archivo(file,std::ios::in);
    if (!archivo) {
		std::cout << "No se pudo abrir el archivo." << std::endl << std::endl;
    } else {
		std::string linea;
        char delimitador = ','; int i=0;
        while (getline(archivo, linea)) {
			std::stringstream stream(linea);
			std::string dato;
            getline(stream,dato,delimitador); //ID
            getline(stream,dato,delimitador); //Maligno o benigno
            if (dato == "M"){Y[i] = {0,1};} //Cambio de M y B a 0 y 1
            else {Y[i] = {1,0};}
			
            for (int j = 0;j <= 29; j++){
                float float1;
                getline(stream,dato,delimitador);
				std::istringstream(dato) >> float1;
                X[i][j] = float1;
            }
            i++;
        }
        archivo.close();
    }

	return {X, Y};
}

std::tuple<Matriz, Matriz, Matriz, Matriz> divideData(const Matriz& x, const Matriz& y, double div){
	if (x.size() != y.size()) {
        throw std::runtime_error("Dimension mismatch");
    }
	if(div > 1 or div < 0){
		throw std::runtime_error("div out of range");
	}
	u32 train_size = x.size()*div;
	return {Matriz(x.begin(), x.begin() + train_size), Matriz(x.begin() + train_size, x.end()),
	Matriz(y.begin(), y.begin() + train_size), Matriz(y.begin() + train_size, y.end())};
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

double tanhActivation(double x) {
    return tanh(x);
}

double tanhDerivative(double x) {
    double tanhx = tanh(x);
    return 1.0 - tanhx * tanhx;
}

double relu(double x) {
    return (x > 0.0) ? x : 0.0;
}

double reluDerivative(double x) {
    return (x > 0.0) ? 1.0 : 0.0;
}


void print(const Lista& lista) {
    std::cout << "[ ";
    for (const auto& val : lista) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
}

void print(const Matriz& matriz) {
    std::cout << "[\n";
    for (const auto& fila : matriz) {
        std::cout << "  ";
        print(fila);
    }
    std::cout << "]" << std::endl;
}

double normal_loss(const Lista& output, const Lista& target){
    if (output.size() != target.size()) {
        throw std::runtime_error("Dimension mismatch");
    }
	double result = 0;
	for(u32 i = 0; i < output.size(); ++i){
		result += (output[i] - target[i])*(output[i] - target[i]);
	}
	return result;
}

double normal_loss_derivative(const Lista& output, const Lista& target){
    if (output.size() != target.size()) {
        throw std::runtime_error("Dimension mismatch");
    }
	double result = 0;
	for(u32 i = 0; i < output.size(); ++i){
		result += 2.0*(output[i] - target[i])*(output[i] - target[i]);
	}
	return result;
}


std::unordered_map<FunctionType, FunctionType> derivative{
	{sigmoid, sigmoidDerivative},
	{tanhActivation, tanhDerivative},
	{relu, reluDerivative}
};


std::vector<double> vector_matrix_multiplication(const Lista& vector, const Matriz& matrix) {
    if (vector.size() != matrix[0].size()) {
		std::cout << "vs: "<< vector.size() << " cm:" << matrix[0].size() << std::endl;
        throw std::runtime_error("Dimension mismatch");
    }

    const u32 num_threads = std::thread::hardware_concurrency();
    const u32 chunk_size = matrix.size() / num_threads;
    std::vector<double> result(matrix.size(), 0.0);

    auto multiply_thread = [&](u32 start, u32 end) {
        for (auto i = start; i < end; ++i) {
            for (u32 j = 0; j < matrix[i].size(); ++j) {
                result[i] += vector[j] * matrix[i][j];
            }
        }
    };

    std::vector<std::thread> threads;
    for (u32 i = 0; i < num_threads; ++i) {
        auto start = i * chunk_size;
        auto end = (i == num_threads - 1) ? matrix.size() : (i + 1) * chunk_size;
        threads.emplace_back(multiply_thread, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return result;
}
//RedNeuronal {{{1
class RedNeuronal {
	public:
		// input: numero de entradas, numero de capas ocultas, numero de neuronas por capa oculta, numero de neuronas en capa de salida, funciones de activacion por capa
		RedNeuronal(u32 ne, u32 nco, const std::vector<u32>& nnco, u32 nncs, const std::vector<FunctionType>& fac, u32 seed = std::random_device{}()):
		m_seed(seed)
	{
			m_matrices.reserve(nco + 1);	
			m_activations.reserve(nco + 1);
			m_delta.reserve(nco + 1);

			m_matrices.push_back(crear_matriz(nnco[0], ne));
			m_activations.push_back(Lista(nnco[0], 0.0));
			m_delta.push_back(Lista(nnco[0], 0.0));
			for(u32 i = 1; i < nco; ++i){
				m_matrices.push_back(crear_matriz(nnco[i], nnco[i - 1]));
				m_activations.push_back(Lista(nnco[i], 0.0));
				m_delta.push_back(Lista(nnco[i], 0.0));
			}
			m_matrices.push_back(crear_matriz(nncs, nnco.back()));
			m_activations.push_back(Lista(nncs, 0.0));
			m_delta.push_back(Lista(nncs, 0.0));

			m_funciones_de_activacion = fac;
		}

		void train(const Matriz& X, const Matriz& Y, u32 epochs, double learning_rate = 0.5f) {
			if (X.size() != Y.size()) {
				throw std::runtime_error("Input and target dimensions do not match.");
			}

			for (u32 epoch = 0; epoch < epochs; ++epoch) {
				// crear estructura con las mismas dimensiones de nuestra matrices
				// llena de 0 para guardar los promedios
				auto gradiente = m_matrices;
				double error_promedio = 0;
				llenar_de_ceros(gradiente);
				for (u32 i = 0; i < X.size(); ++i) {
					llenar_de_ceros(m_delta);
					auto result = forward(X[i]);
					error_promedio += normal_loss(result, Y[i]);

					// computar neuronas del ultimo layer
					for(u32 j = 0; j < result.size(); ++j){
						m_delta[m_delta.size() -1][j] = 2*(result[j] - Y[i][j])*result[j]*(1 - result[j]);
					}

					// computar para el resto de layers
					for(u32 l = m_delta.size() - 2; l > 0; --l){
						//por cada neurona del layer actual

						for(u32 j = 0; j < m_delta[l].size(); ++j){
							double sum = 0;
							// por cada neurona del siguiente layer
							
							for(u32 k = 0; k < m_delta[l + 1].size(); ++k){
								sum += m_delta[l + 1][k]*m_matrices[l + 1][k][j]*m_activations[l][j]*(1 - m_activations[l][j]);
							}
							m_delta[l][j] = sum;
						}
					}

					for(u32 j = 0; j < gradiente[0].size(); ++j){
						for(u32 k = 0; k < X[i].size(); ++k){
							gradiente[0][j][k] += m_delta[0][j]*X[i][k];
						}
					}

					// computar la gradiente de este x
					for(u32 l = 1; l < gradiente.size(); ++l){
						// por cada nodo
						for(u32 j = 0; j < gradiente[l].size(); ++j){
							// por cada peso
							for(u32 k = 0; k < gradiente[l][j].size(); ++k){
								gradiente[l][j][k] += m_delta[l][j] * m_activations[l - 1][k];
							}
						}
					}
				}
				//calcular el promedio de la gradiente
				
				for(u32 l = 0; l < gradiente.size(); ++l){
					// por cada nodo
					for(u32 j = 0; j < gradiente[l].size(); ++j){
						// por cada peso
						for(u32 k = 0; k < gradiente[l][j].size(); ++k){
							gradiente[l][j][k] /= X.size();
						}
					}
				}
				//gradient decend

				for(u32 l = 0; l < gradiente.size(); ++l){
					// por cada nodo
					for(u32 j = 0; j < gradiente[l].size(); ++j){
						// por cada peso
						for(u32 k = 0; k < gradiente[l][j].size(); ++k){
							m_matrices[l][j][k] -= learning_rate*gradiente[l][j][k];
						}
					}
				}
				std::cout << epoch << " " << error_promedio / X.size() << std::endl;
			}
		}

		Lista predict(const Lista& X){
			auto result = X;
			for(u32 i = 0; i < m_matrices.size(); ++i){
				result = vector_matrix_multiplication(result, m_matrices[i]);
				result = aplicar_activacion(result, m_funciones_de_activacion[i]);
			}
			return result;
		}

	private:
		Matriz crear_matriz(u32 rows, u32 columns) {
			std::mt19937 gen(m_seed++);
			std::uniform_real_distribution<double> dis(0.0, 1.0);

			Matriz matriz(rows, Lista(columns, 0.0));

			for (auto& fila : matriz) {
				for (auto& elemento : fila) {
					elemento = dis(gen);
				}
			}


			return matriz;
		}

		Lista forward(const Lista& input){
			auto result = input;
			for(u32 i = 0; i < m_matrices.size(); ++i){
				result = vector_matrix_multiplication(result, m_matrices[i]);
				result = aplicar_activacion(result, m_funciones_de_activacion[i]);
				m_activations[i] = result;
			}
			return result;
		}
		
		Lista aplicar_activacion(const Lista& input, FunctionType funcion){
			Lista result = input;
			for(auto& val: result){
				val = funcion(val);
			}

			return result;
		}
		
		Lista elemento_por_elemento_multiplication(const Lista& a, const Lista& b) {
			Lista result;
			for (u32 i = 0; i < a.size(); ++i) {
				result.push_back(a[i] * b[i]);
			}
			return result;
		}


    Lista aplicar_activacion_derivada(const Lista& input, FunctionType funcion) {
        Lista result = input;
        for (auto& val : result) {
            val = derivative[funcion](val);
        }

        return result;
    }

    Matriz transpose(const Matriz& matrix) {
        Matriz result(matrix[0].size(), Lista(matrix.size(), 0.0));

        for (u32 i = 0; i < matrix.size(); ++i) {
            for (u32 j = 0; j < matrix[i].size(); ++j) {
                result[j][i] = matrix[i][j];
            }
        }

        return result;
    }
	void llenar_de_ceros(std::vector<Matriz>& vector_matrices) {
		for (auto& matriz : vector_matrices) {
			for (auto& fila : matriz) {
				for (auto& elemento : fila) {
					elemento = 0.0;
				}
			}
		}
	}

	void llenar_de_ceros(std::vector<Lista>& vector_Lista) {
		for (auto& lista : vector_Lista) {
				for (auto& elemento : lista) {
					elemento = 0.0;
				}
		}
	}

		u32 m_seed;		
		std::vector<Matriz> m_matrices;
		std::vector<FunctionType> m_funciones_de_activacion;
		std::vector<Lista> m_activations;
		std::vector<Lista> m_delta;
};
// Main {{{1
int main(){

	RedNeuronal nn_sigmoid(30, 2, {10, 2}, 2, {sigmoid, sigmoid, sigmoid}, 8);
	RedNeuronal nn_tanh(30, 2, {10, 2}, 2, {tanhActivation, tanhActivation, tanhActivation}, 8);
	RedNeuronal nn_relu(30, 2, {10, 2}, 2, {relu, relu, relu}, 8);

	auto [X, Y] = loadData();

    // Crear un vector de índices
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0); // Llenar con 0, 1, 2, ..., X.size()-1
	
    std::random_device rd;
    std::mt19937 gen(rd());

    // Barajar los índices
    std::shuffle(indices.begin(), indices.end(), gen);

    // Aplicar el shuffle a X e Y
    Matriz X_shuffled(X.size());
    Matriz Y_shuffled(Y.size());
	
    for (size_t i = 0; i < X.size(); ++i) {	
        X_shuffled[i] = X[indices[i]];
		Y_shuffled[i] = Y[indices[i]];
    }
	auto [X_train, X_test, Y_train, Y_test] = divideData(X_shuffled, Y_shuffled, 0.7);	
	
	std::cout << "Sigmoid\n";
	nn_sigmoid.train(X_train, Y_train, 1000, 0.01);

	std::cout << "ReLu\n";
	nn_relu.train(X_train, Y_train, 1000, 0.01);	

	std::cout << "Tanh\n";
	nn_tanh.train(X_train, Y_train, 1000, 0.01);

	return 0;
}
