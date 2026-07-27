using System;
using System.IO;
using UnityEngine; // Adaugam Unity

public class NeuralNetworkInference : MonoBehaviour // Mostenim din MonoBehaviour
{
    // Layer 1
    private double[,] W1;
    private double[] b1;
    // Layer 2
    private double[,] W2;
    private double[] b2;

    public bool IsLoaded { get; private set; } = false;

    // Numele fisierului din StreamingAssets
    public string modelName = "model.bin"; 

    // Se apeleaza automat cand porneste jocul
    void Start()
    {
        // Cauta <modelName> in folderul StreamingAssets
        string path = Path.Combine(Application.streamingAssetsPath, modelName);
        LoadModel(path);
    }

    public void LoadModel(string path)
    {
        if (!File.Exists(path))
        {
            Debug.LogError($"Error: Model file not found at {path}");
            return;
        }

        try 
        {
            using (var reader = new BinaryReader(File.Open(path, FileMode.Open)))
            {
                W1 = ReadMatrix(reader);
                b1 = ReadVector(reader);
                W2 = ReadMatrix(reader);
                b2 = ReadVector(reader);
            }
            IsLoaded = true;
            Debug.Log("Neural Network Model Loaded Successfully!");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to load model: {e.Message}");
        }
    }

    public int Predict(double[] input)
    {
        if (!IsLoaded) return -1;

        // Forward Prop (Layer 1)
        double[] Z1 = Dense(W1, input, b1);
        double[] A1 = ReLU(Z1);

        // Forward Prop (Layer 2)
        double[] Z2 = Dense(W2, A1, b2);
        double[] A2 = Softmax(Z2);

        return ArgMax(A2);
    }

    // --- Helpers (No External Libraries) ---

    private double[] Dense(double[,] W, double[] x, double[] b)
    {
        int rows = W.GetLength(0);
        int cols = W.GetLength(1);
        double[] output = new double[rows];

        // Parallel.For can speed this up in Unity, but standard loop is fine for small nets
        for (int i = 0; i < rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < cols; j++)
            {
                sum += W[i, j] * x[j];
            }
            output[i] = sum + b[i];
        }
        return output;
    }

    private double[] ReLU(double[] z)
    {
        double[] a = new double[z.Length];
        for (int i = 0; i < z.Length; i++)
        {
            a[i] = Math.Max(0, z[i]);
        }
        return a;
    }

    private double[] Softmax(double[] z)
    {
        double max = double.MinValue;
        foreach (var val in z) if (val > max) max = val;

        double sum = 0;
        double[] exps = new double[z.Length];
        for (int i = 0; i < z.Length; i++)
        {
            exps[i] = Math.Exp(z[i] - max); // Stability trick
            sum += exps[i];
        }

        for (int i = 0; i < z.Length; i++)
        {
            exps[i] /= sum;
        }
        return exps;
    }

    private int ArgMax(double[] a)
    {
        int maxIndex = 0;
        double maxValue = a[0];
        for (int i = 1; i < a.Length; i++)
        {
            if (a[i] > maxValue)
            {
                maxValue = a[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    // --- Readers matching the format saved in Console App ---
    private double[,] ReadMatrix(BinaryReader reader)
    {
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        double[,] m = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                m[i, j] = reader.ReadDouble();
            }
        }
        return m;
    }

    private double[] ReadVector(BinaryReader reader)
    {
        // Vectors were saved as Matrices (rows x 1)
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32(); // Should be 1
        double[] v = new double[rows];
        for (int i = 0; i < rows; i++)
        {
            // Consumăm coloana (care e 1)
            for (int j = 0; j < cols; j++) 
            {
                double val = reader.ReadDouble();
                if (j == 0) v[i] = val;
            }
        }
        return v;
    }
}
