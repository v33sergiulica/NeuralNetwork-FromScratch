using System;
using System.IO;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

public class DataReader
{
    public static double[][] ReadImages(string path)
    {
        using (BinaryReader br = new BinaryReader(File.OpenRead(path)))
        {
            int magic = ReadBigEndianInt32(br);
            int numImages = ReadBigEndianInt32(br);
            int rows = ReadBigEndianInt32(br);
            int cols = ReadBigEndianInt32(br);

            double[][] images = new double[numImages][];
            for (int i = 0; i < numImages; i++)
            {
                images[i] = new double[rows * cols];
                for (int p = 0; p < rows * cols; p++)
                {
                    images[i][p] = br.ReadByte() / 255.0;
                }
            }
            return images;
        }
    }

    public static byte[] ReadLabels(string path)
    {
        using (BinaryReader br = new BinaryReader(File.OpenRead(path)))
        {
            int magic = ReadBigEndianInt32(br);
            int numLabels = ReadBigEndianInt32(br);
            return br.ReadBytes(numLabels);
        }
    }

    public static double[][] AugmentData(double[][] images, int duplicatesPerImage)
    {
        Random rng = new Random();
        int n = images.Length;
        // Total images = original (n) + duplicates (n * duplicatesPerImage)
        double[][] augmented = new double[n * (duplicatesPerImage + 1)][];
        
        // Parallel.For pentru viteza
        System.Threading.Tasks.Parallel.For(0, n, i =>
        {
            augmented[i] = images[i]; // Pastram originalul
            for (int j = 0; j < duplicatesPerImage; j++)
            {
                augmented[n + i * duplicatesPerImage + j] = ApplyAugmentation(images[i], i * 100 + j); // Seed deterministic pt debug
            }
        });
        
        return augmented;
    }

    public static byte[] AugmentLabels(byte[] labels, int duplicatesPerImage)
    {
        int n = labels.Length;
        byte[] augmented = new byte[n * (duplicatesPerImage + 1)];
        
        for (int i = 0; i < n; i++)
        {
            augmented[i] = labels[i];
            for (int j = 0; j < duplicatesPerImage; j++)
            {
                augmented[n + i * duplicatesPerImage + j] = labels[i];
            }
        }
        return augmented;
    }

    private static double[] ApplyAugmentation(double[] img, int seed)
    {
        Random rng = new Random(seed); // Thread-safe random

        // 1. Translatie (+/- 2 pixeli) - User a cerut "putin"
        int shiftX = rng.Next(-2, 3);
        int shiftY = rng.Next(-2, 3);
        double[] translated = Translate(img, shiftX, shiftY);

        // 2. Scalare (0.85x .. 1.15x)
        double scale = 0.85 + rng.NextDouble() * 0.3;
        double[] scaled = Scale(translated, scale);

        // 3. Rotatie (+/- 15 grade)
        double angle = (rng.NextDouble() - 0.5) * 30;
        double[] rotated = Rotate(scaled, angle);
        
        // 4. Zgomot
        return AddNoise(rotated, rng);
    }

    private static double[] Translate(double[] img, int shiftX, int shiftY)
    {
        double[] result = new double[784];
        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                int newX = x - shiftX; // Shift logic invers pentru a muta imaginea
                int newY = y - shiftY;
                
                if (newX >= 0 && newX < 28 && newY >= 0 && newY < 28)
                {
                    result[y * 28 + x] = img[newY * 28 + newX];
                }
            }
        }
        return result;
    }

    private static double[] Scale(double[] img, double factor)
    {
        double[] result = new double[784];
        double centerX = 13.5;
        double centerY = 13.5;

        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                // Coordonatele in imaginea sursa (inversul scalarii)
                double srcX = (x - centerX) / factor + centerX;
                double srcY = (y - centerY) / factor + centerY;

                // Bilinear interpolation simplificat (Nearest Neighbor pt viteza)
                int sx = (int)Math.Round(srcX);
                int sy = (int)Math.Round(srcY);

                if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28)
                {
                    result[y * 28 + x] = img[sy * 28 + sx];
                }
            }
        }
        return result;
    }

    private static double[] Rotate(double[] img, double angleDegrees)
    {
        double rad = angleDegrees * Math.PI / 180.0;
        double cos = Math.Cos(rad);
        double sin = Math.Sin(rad);
        double[] result = new double[784];
        double centerX = 13.5;
        double centerY = 13.5;

        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                // Rotatie inversa in jurul centrului
                double dx = x - centerX;
                double dy = y - centerY;
                
                double srcX = centerX + (dx * cos - dy * sin);
                double srcY = centerY + (dx * sin + dy * cos);

                int sx = (int)Math.Round(srcX);
                int sy = (int)Math.Round(srcY);

                if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28)
                {
                    result[y * 28 + x] = img[sy * 28 + sx];
                }
            }
        }
        return result;
    }

    private static double[] AddNoise(double[] img, Random rng)
    {
        double[] result = new double[784];
        for (int i = 0; i < 784; i++)
        {
            double noise = (rng.NextDouble() - 0.5) * 0.1; // +/- 0.05
            result[i] = Math.Clamp(img[i] + noise, 0.0, 1.0);
        }
        return result;
    }

    private static int ReadBigEndianInt32(BinaryReader br)
    {
        var bytes = br.ReadBytes(4);
        Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }

    public static void Shuffle(double[][] images, byte[] labels)
    {
        Random rng = new Random();
        int n = images.Length;
        while (n > 1)
        {
            n--;
            int k = rng.Next(n + 1);
            
            double[] tempImage = images[k];
            images[k] = images[n];
            images[n] = tempImage;
            byte tempLabel = labels[k];
            labels[k] = labels[n];
            labels[n] = tempLabel;
        }
    }
}

public class NeuralNetwork {
    public Matrix<double> W1, W2;
    public Matrix<double> b1, b2;

    private static double ReLU(double x) {
        return Math.Max(0, x);
    }

    private static double ReLU_derivative(double x) {
        return x > 0 ? 1 : 0;
    }

    private static Matrix<double> Softmax(Matrix<double> z)
    {
        var result = Matrix<double>.Build.Dense(z.RowCount, z.ColumnCount);
        for (int i = 0; i < z.ColumnCount; i++)
        {
            var col = z.Column(i);
            double max = col.Maximum();
            var exps = col.Subtract(max).PointwiseExp();
            double sum = exps.Sum();
            result.SetColumn(i, exps.Divide(sum));
        }
        return result;
    }
    
    public void init_params() {
        Random rng = new Random();
        W1 = Matrix<double>.Build.Dense(32, 784, (i, j) => (rng.NextDouble() - 0.5) * 0.02); 
        b1 = Matrix<double>.Build.Dense(32, 1, (i, j) => (rng.NextDouble() - 0.5) * 0.02);
        W2 = Matrix<double>.Build.Dense(10, 32, (i, j) => (rng.NextDouble() - 0.5) * 0.02);
        b2 = Matrix<double>.Build.Dense(10, 1, (i, j) => (rng.NextDouble() - 0.5) * 0.02);
    }

    public static Matrix<double> AddBias(Matrix<double> m, Matrix<double> b)
    {
        var result = Matrix<double>.Build.Dense(m.RowCount, m.ColumnCount);
        for (int j = 0; j < m.ColumnCount; j++)
        {
            result.SetColumn(j, (m.Column(j) + b.Column(0)));
        }
        return result;
    }

    public (Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>) forward_prop(Matrix<double> W1, Matrix<double> b1, Matrix<double> W2, Matrix<double> b2, Matrix<double> X) {
        Matrix<double> Z1 = AddBias(W1 * X, b1); 
        Matrix<double> A1 = Z1.Map(ReLU);
        Matrix<double> Z2 = AddBias(W2 * A1, b2);
        Matrix<double> A2 = Softmax(Z2);
        return (Z1, A1, Z2, A2);
    }

    public static Matrix<double> one_hot(byte[] labels) 
    {
        var oneHot = Matrix<double>.Build.Dense(10, labels.Length); 
        for (int i = 0; i < labels.Length; i++)
        {
            int label = labels[i]; 
            oneHot[label, i] = 1.0;
        }
        return oneHot;
    }

    public static (Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>) backward_prop(Matrix<double> Z1, Matrix<double> A1, Matrix<double> Z2, Matrix<double> A2, Matrix<double> W2, Matrix<double> X, byte[] Y) 
    {
        int m = X.ColumnCount; 
        Matrix<double> one_hot_Y = one_hot(Y);
        Matrix<double> dZ2 = A2 - one_hot_Y;
        Matrix<double> dW2 = (1.0 / m) * dZ2 * A1.Transpose();
        Matrix<double> db2 = (1.0 / m) * dZ2.RowSums().ToColumnMatrix();
        Matrix<double> dZ1 = (W2.Transpose() * dZ2).PointwiseMultiply(Z1.Map(ReLU_derivative));
        Matrix<double> dW1 = (1.0 / m) * dZ1 * X.Transpose();
        Matrix<double> db1 = (1.0 / m) * dZ1.RowSums().ToColumnMatrix();
        return (dW1, db1, dW2, db2);
    }

    public (Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>) update_params(Matrix<double> W1, Matrix<double> b1, Matrix<double> W2, Matrix<double> b2, Matrix<double> dW1, Matrix<double> db1, Matrix<double> dW2, Matrix<double> db2, double learning_rate) 
    {
        W1 = W1 - learning_rate * dW1;
        b1 = b1 - learning_rate * db1;
        W2 = W2 - learning_rate * dW2;
        b2 = b2 - learning_rate * db2;
        return (W1, b1, W2, b2);
    }

    public static int[] get_predictions(Matrix<double> A2) 
    {
        int m = A2.ColumnCount;
        int[] predictions = new int[m];
        for (int i = 0; i < m; i++) 
        {
            predictions[i] = A2.Column(i).MaximumIndex();
        }
        return predictions;
    }

    public static double get_accuracy(int[] predictions, byte[] Y) 
    {
        int correct = 0;
        for (int i = 0; i < Y.Length; i++) 
        {
            if (predictions[i] == Y[i]) 
            {
                correct++;
            }
        }
        return (double)correct / Y.Length;
    }

    public void gradient_descent(Matrix<double> X, byte[] Y, int epochs, double learning_rate, int batch_size = 128) 
    {
        init_params();
        int m = X.ColumnCount;
        int num_batches = (int)Math.Ceiling((double)m / batch_size);

        for (int epoch = 0; epoch < epochs; epoch++) 
        {
            double total_loss = 0;
            int correct_preds = 0;

            for (int batch = 0; batch < num_batches; batch++)
            {
                int start = batch * batch_size;
                int end = Math.Min(start + batch_size, m);
                int current_batch_size = end - start;

                // Nu putem face slice usor in MathNet pe coloane, asa ca folosim SubMatrix
                // SubMatrix(rowIndex, rowCount, columnIndex, columnCount)
                Matrix<double> X_batch = X.SubMatrix(0, X.RowCount, start, current_batch_size);
                
                // Extragem Y_batch manual
                byte[] Y_batch = new byte[current_batch_size];
                Array.Copy(Y, start, Y_batch, 0, current_batch_size);

                var (Z1, A1, Z2, A2) = forward_prop(W1, b1, W2, b2, X_batch);
                var (dW1, db1, dW2, db2) = backward_prop(Z1, A1, Z2, A2, W2, X_batch, Y_batch);
                (W1, b1, W2, b2) = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate);
                
                // Monitorizare (optional)
                int[] preds = get_predictions(A2);
                for(int k=0; k<current_batch_size; k++) if(preds[k] == Y_batch[k]) correct_preds++;
            }

            // Decay learning rate
            if (epoch % 5 == 0 && epoch > 0) learning_rate *= 0.9;

            double epoch_acc = (double)correct_preds / m;
            Console.WriteLine($"Epoca {epoch + 1}/{epochs}: Acuratete: {epoch_acc:P2} (LR: {learning_rate:F4})");
        }
    }

    public void SaveModel(string path)
    {
        using (var writer = new BinaryWriter(File.Open(path, FileMode.Create)))
        {
            WriteMatrix(writer, W1);
            WriteMatrix(writer, b1);
            WriteMatrix(writer, W2);
            WriteMatrix(writer, b2);
        }
        Console.WriteLine($"Model salvat in {path}");
    }

    public void LoadModel(string path)
    {
        using (var reader = new BinaryReader(File.Open(path, FileMode.Open)))
        {
            W1 = ReadMatrix(reader);
            b1 = ReadMatrix(reader);
            W2 = ReadMatrix(reader);
            b2 = ReadMatrix(reader);
        }
        Console.WriteLine($"Model incarcat din {path}");
    }

    private void WriteMatrix(BinaryWriter writer, Matrix<double> m)
    {
        writer.Write(m.RowCount);
        writer.Write(m.ColumnCount);
        for (int i = 0; i < m.RowCount; i++)
        {
            for (int j = 0; j < m.ColumnCount; j++)
            {
                writer.Write(m[i, j]);
            }
        }
    }

    private Matrix<double> ReadMatrix(BinaryReader reader)
    {
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();
        var m = Matrix<double>.Build.Dense(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                m[i, j] = reader.ReadDouble();
            }
        }
        return m;
    }
}

class Program
{
    public static void Main(string[] args)
    {
        string trainImagesPath = "train-images.idx3-ubyte";
        string trainLabelsPath = "train-labels.idx1-ubyte";

        if (!File.Exists(trainImagesPath) || !File.Exists(trainLabelsPath))
        {
            Console.WriteLine("Lipsesc fisierele MNIST!");
            return;
        }

        Console.WriteLine("Citesc datele...");
        double[][] images = DataReader.ReadImages(trainImagesPath);
        byte[] labels = DataReader.ReadLabels(trainLabelsPath);

        Console.WriteLine($"Original: {images.Length} imagini.");
        
        // --- DATA AUGMENTATION ---
        Console.WriteLine("Augmenting data (x2)...");
        images = DataReader.AugmentData(images, 1); // 1 duplicat = x2 total
        labels = DataReader.AugmentLabels(labels, 1);
        
        // Shuffle este CRUCIAL dupa augmentare pentru a amesteca originalele cu copiile
        Console.WriteLine("Shuffling data...");
        DataReader.Shuffle(images, labels);

        Console.WriteLine($"Total dupa augmentare: {images.Length} imagini.");
        Console.WriteLine("Convertesc datele in format matriceal.");
        int rows = 784;
        int m = images.Length;
        Matrix<double> X = Matrix<double>.Build.DenseOfColumnArrays(images); 
        
        Console.WriteLine("\n    Start Antrenament");
        NeuralNetwork nn = new NeuralNetwork();
        
        string modelPath = "model_augmented.bin"; // Nume nou pentru modelul augmentat
        if (File.Exists(modelPath))
        {
            Console.WriteLine($"\nModel antrenat gasit ({modelPath})! Il incarc.");
            nn.LoadModel(modelPath);
        }
        else
        {
            Console.WriteLine("\nNu exista model salvat. Incep antrenarea.");
            // Antrenam pe datele augmentate (Mini-Batch)
            // 20 de epoci e mai mult decat suficient (fiecare epoca vede toate datele)
            // batch_size implicit e 128
            nn.gradient_descent(X, labels, 20, 0.5); 
            nn.SaveModel(modelPath);
        }

        Console.WriteLine("\n    Testare pe setul de test (10k imagini)");
        string testImagesPath = "t10k-images.idx3-ubyte";
        string testLabelsPath = "t10k-labels.idx1-ubyte";
        
        if (File.Exists(testImagesPath) && File.Exists(testLabelsPath))
        {
            var testImages = DataReader.ReadImages(testImagesPath);
            var testLabels = DataReader.ReadLabels(testLabelsPath);
            var X_test = Matrix<double>.Build.DenseOfColumnArrays(testImages);            
            var (_, _, _, A2_test) = nn.forward_prop(nn.W1, nn.b1, nn.W2, nn.b2, X_test);
            int[] predictions = NeuralNetwork.get_predictions(A2_test);
            double accuracy = NeuralNetwork.get_accuracy(predictions, testLabels);
            Console.WriteLine($"Acuratete pe TEST: {accuracy:P2}");
        }
        else
        {
             Console.WriteLine("Nu am gasit fisierele de test t10k-");
        }

        Console.WriteLine("\nExecutie completa!");
        Console.WriteLine("Apasa orice tasta pentru a iesi.");
        Console.ReadKey();
    }
}
