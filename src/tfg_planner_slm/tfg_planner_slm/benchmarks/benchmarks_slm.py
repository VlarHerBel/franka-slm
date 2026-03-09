import requests
import json
import time
import pandas as pd

# 1. Configuración
MODELS = ["phi3", "gemma2:2b", "qwen2.5:1.5b"]
URL = "http://localhost:11434/api/generate"

# 2. El Prompt de Sistema (Crucial para tu TFG)
# Aquí definimos que el modelo NO debe hablar, solo escupir JSON.
SYSTEM_PROMPT = """
You are a robotic planner for a Franka Emika manipulator.
Your goal is to convert natural language commands into a JSON sequence.
The output must contain ONLY the JSON structure, no explanations.
Valid actions: PICK, PLACE, PUSH.
Example format: {"plan": [{"action": "PICK", "object": "red_box"}]}
"""

# 3. Pruebas (Instrucciones del usuario)
TEST_PROMPTS = [
    "Pick up the blue cube and place it on the table.",
    "Push the screw away and grasp the nut.",
    "Move the red block to position A and the green block to position B."
]

results = []

print(f"--- INICIANDO BENCHMARK SLM PARA ROBÓTICA ---")

for model in MODELS:
    print(f"\nProbando modelo: {model}...")
    
    for i, prompt in enumerate(TEST_PROMPTS):
        payload = {
            "model": model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "format": "json"  # Forzamos modo JSON de Ollama (si el modelo lo soporta)
        }
        
        start_time = time.time()
        try:
            response = requests.post(URL, json=payload).json()
            end_time = time.time()
            
            # Métricas
            latency = end_time - start_time
            content = response.get("response", "")
            eval_count = response.get("eval_count", 0) # Número de tokens generados
            tps = eval_count / latency if latency > 0 else 0 # Tokens por segundo
            
            # Validación simple de JSON
            valid_json = False
            try:
                json.loads(content)
                valid_json = True
            except:
                valid_json = False

            results.append({
                "Model": model,
                "Test_ID": i,
                "Latency (s)": round(latency, 3),
                "Tokens/Sec": round(tps, 2),
                "Valid JSON": valid_json,
                "Output": content[:100] + "..." # Guardamos un trozo para ver
            })
            
        except Exception as e:
            print(f"Error con {model}: {e}")

# 4. Mostrar Resultados
df = pd.DataFrame(results)
print("\n--- RESULTADOS COMPARATIVOS ---")
print(df.to_markdown())

# Opcional: Guardar a CSV para tu memoria del TFG
df.to_csv("slm_benchmark_results.csv", index=False)