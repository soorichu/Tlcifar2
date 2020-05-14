package co.soori.tlcifar2;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.RectF;
import android.os.Build;
import android.os.Bundle;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.nio.MappedByteBuffer;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.provider.MediaStore;
import android.util.Log;
import android.view.InputDevice;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;
import java.io.InputStream;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class MainActivity extends AppCompatActivity {

    private ImageView image;
    private TextView textView;

    private static final int PERMISSIONS_REQUEST_CODE = 1000;
    String[] PERMISSIONS  = {Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE};

    private static final int REQUEST_IMAGE_CAPTURE = 1;

    private Interpreter tflite;
    private static final String MODEL_FILE = "CIFAL10.tflite";

    private enum Device { CPU, NNAPI, GPU }

    public String argmax(Map<String, Float> map){
        Map.Entry<String, Float> maxEntry = null;
        for(Map.Entry<String, Float> entry : map.entrySet()){
            if(maxEntry==null || entry.getValue().compareTo(maxEntry.getValue())>0){
                maxEntry = entry;
            }
        }
        return maxEntry.getKey();
    }

    public static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename) throws IOException
    {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getDeclaredLength();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    public void imageProcess(Bitmap bitmap){
        MappedByteBuffer tfliteModel;
        List<String> labels = new ArrayList<>(10);
        labels.add("비행기");
        labels.add("자동차");
        labels.add("새");
        labels.add("고양이");
        labels.add("사슴");
        labels.add("개");
        labels.add("개구리");
        labels.add("말");
        labels.add("배");
        labels.add("트럭");
        NnApiDelegate nnApiDelegate = null;  Log.e("imageProcess", "1");
        int[] deviceId = InputDevice.getDeviceIds();  Log.e("imageProcess", "2");
        InputDevice device = InputDevice.getDevice(deviceId[0]);  Log.e("imageProcess", "3");
        Interpreter.Options tfliteOptions = new Interpreter.Options();  Log.e("imageProcess", "4");
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P){  Log.e("imageProcess", "5");
            nnApiDelegate = new NnApiDelegate();  Log.e("imageProcess", "6");
            tfliteOptions.addDelegate(nnApiDelegate);  Log.e("imageProcess", "7");
        }
        try {
            tfliteModel = FileUtil.loadMappedFile(this, MODEL_FILE);  Log.e("imageProcess", "8");
            tflite = new Interpreter(tfliteModel, tfliteOptions);  Log.e("imageProcess", "9");
   //         labels = FileUtil.loadLabels(this, "labels.txt");

        }catch(IOException e){
            e.printStackTrace();
        }

        int imageTensorIndex = 0;  Log.e("imageProcess", "10");
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape();  Log.e("imageProcess", "11");  //{1, 32, 32, 3}
        int imageSizeY = imageShape[1], imageSizeX = imageShape[0];  Log.e("imageProcess", "12");
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();  Log.e("imageProcess", "13");
        int probabilityTensorIndex = 0;  Log.e("imageProcess", "");
        int[] probabilityShape = tflite.getOutputTensor(probabilityTensorIndex).shape();  Log.e("imageProcess", "14"); //{1, NUM_CLASSES}
        DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();  Log.e("imageProcess", "15");

        TensorImage inputImageBuffer = new TensorImage(imageDataType);  Log.e("imageProcess", "16");
        TensorBuffer outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);  Log.e("imageProcess", "17");
        TensorProcessor probabilityProcessor = new TensorProcessor.Builder().add(new NormalizeOp(.0f, 1.f)).build();  Log.e("imageProcess", "18");

        inputImageBuffer.load(bitmap);  Log.e("imageProcess", "19");
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());  Log.e("imageProcess", "20");
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(new Rot90Op(0))
                        .add(new NormalizeOp(0.47f, 0.25f))
                        .build();  Log.e("imageProcess", "21");
        inputImageBuffer = imageProcessor.process(inputImageBuffer);  Log.e("imageProcess", "22");
        tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());  Log.e("imageProcess", "23");

        Map<String, Float> labledProbability = new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer)).getMapWithFloatValue();  Log.e("imageProcess", "24");
        textView = (TextView) findViewById(R.id.result);  Log.e("imageProcess", "25");
        textView.setText(argmax(labledProbability));  Log.e("imageProcess", "26");

        tflite.close();  Log.e("imageProcess", "27");
        if(nnApiDelegate != null){  Log.e("imageProcess", "28");
            nnApiDelegate.close();  Log.e("imageProcess", "29");
        }

    }


    public void takePicture(View view){
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            try {
                Bitmap imageBitmap = (Bitmap) extras.get("data");
                image = (ImageView) findViewById(R.id.imageView);
                image.setImageBitmap(imageBitmap);
                textView = (TextView) findViewById(R.id.result);
                imageProcess(imageBitmap);
            }catch(Exception e){
                e.getStackTrace();
            }
        }

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        System.out.println("model loaded successfully");

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (!hasPermissions(PERMISSIONS)) {
                requestPermissions(PERMISSIONS, PERMISSIONS_REQUEST_CODE);
            }
        }

    }

    private boolean hasPermissions(String[] permissions) {
        int result;
        for (String perms : permissions){
            result = ContextCompat.checkSelfPermission(this, perms);
            if (result == PackageManager.PERMISSION_DENIED){
                return false;
            }
        }
        return true;
    }

}