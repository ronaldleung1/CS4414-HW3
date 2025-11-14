#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <llama.h>

using json = nlohmann::json;

class BGEEncoder {
private:
    llama_model* model;
    llama_context* ctx;
    const llama_vocab* vocab;
    int n_embd;
    
public:
    BGEEncoder(const std::string& model_path) {
        // Initialize backend
        ggml_backend_load_all();
        
        // Load model
        llama_model_params model_params = llama_model_default_params();
        model = llama_load_model_from_file(model_path.c_str(), model_params);
        
        if (model == nullptr) {
            throw std::runtime_error("Failed to load model from: " + model_path);
        }
        
        // Get vocab
        vocab = llama_model_get_vocab(model);
        
        // Get embedding dimension
        n_embd = llama_n_embd(model);
        std::cout << "Model loaded. Embedding dimension: " << n_embd << std::endl;
        
        if (n_embd != 768) {
            std::cerr << "Warning: Expected embedding dimension 768, got " << n_embd << std::endl;
        }
        
        // Create context for embeddings
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 512;  // Context size for BGE
        ctx_params.n_batch = 512;
        ctx_params.embeddings = true;  // Enable embedding mode
        ctx_params.no_perf = false;
        
        ctx = llama_init_from_model(model, ctx_params);
        
        if (ctx == nullptr) {
            llama_model_free(model);
            throw std::runtime_error("Failed to create context");
        }
        
        // Check if this is an encoder model
        if (!llama_model_has_encoder(model)) {
            std::cerr << "Warning: Model does not appear to be an encoder model" << std::endl;
        }
    }
    
    ~BGEEncoder() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
    
    std::vector<float> encode(const std::string& text) {
        // Find number of tokens
        const int n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
        
        if (n_tokens <= 0) {
            throw std::runtime_error("Failed to tokenize text");
        }
        
        // Tokenize the text
        std::vector<llama_token> tokens(n_tokens);
        if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true) < 0) {
            throw std::runtime_error("Failed to tokenize text");
        }
        
        // Create batch using the simpler API
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        
        // Encode using llama_encode
        if (llama_encode(ctx, batch)) {
          throw std::runtime_error("Failed to encode batch");
        }
        
        // Get embeddings
        const float* embd = llama_get_embeddings_seq(ctx, 0);
        
        if (embd == nullptr) {
            throw std::runtime_error("Failed to get embeddings");
        }
        
        std::vector<float> embedding(embd, embd + n_embd);
        
        return embedding;
    }
    
    int get_embedding_dim() const {
        return n_embd;
    }
};

int main(int argc, char** argv) {
    try {
        std::string model_path = "bge-base-en-v1.5-f32.gguf";
        std::string input_file = "documents.json";
        std::string output_file = "preprocessed_documents.json";
        
        std::cout << "Loading documents from " << input_file << "..." << std::endl;
        
        // read input file
        std::ifstream in_file(input_file);
        if (!in_file.is_open()) {
            std::cerr << "Error: Could not open " << input_file << std::endl;
            return 1;
        }
        
        json input_data;
        in_file >> input_data;
        in_file.close();
        
        std::cout << "Found " << input_data.size() << " documents" << std::endl;
        
        // Initialize encoder
        std::cout << "Loading BGE model from " << model_path << "..." << std::endl;
        BGEEncoder encoder(model_path);
        
        // Process documents
        json output_data = json::array();
        
        for (size_t i = 0; i < input_data.size(); i++) {
            auto& doc = input_data[i];
            
            if (i % 100 == 0) {
                std::cout << "Processing document " << i << "/" << input_data.size() << "..." << std::endl;
            }
            
            // Extract text
            std::string text = doc["text"].get<std::string>();
            int id = doc["id"].get<int>();
            
            // Encode to embedding
            std::vector<float> embedding = encoder.encode(text);
            
            // Create output entry
            json entry;
            entry["id"] = id;
            entry["text"] = text;
            entry["embedding"] = embedding;
            
            output_data.push_back(entry);
        }
        
        // Write output JSON
        std::cout << "Writing results to " << output_file << "..." << std::endl;
        std::ofstream out_file(output_file);
        if (!out_file.is_open()) {
            std::cerr << "Error: Could not open " << output_file << " for writing" << std::endl;
            return 1;
        }
        
        out_file << output_data.dump(2);  // Pretty print with 2-space indent
        out_file.close();
        
        std::cout << "Successfully processed " << output_data.size() << " documents" << std::endl;
        std::cout << "Output saved to " << output_file << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

